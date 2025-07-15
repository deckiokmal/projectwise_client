# utils/pipeline_docgen.py
from __future__ import annotations
import asyncio
import json
import traceback
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

from mcp_client.utils.prompt_loader import load_prompt

try:
    _SYSTEM_PROMPT = load_prompt("document_generator").strip()
except Exception:
    _SYSTEM_PROMPT = (
        'Anda adalah "ProjectWise", asisten virtual untuk tim Presales & '
        "Project Manager. Tugas Anda adalah generate document docx berdasarkan "
        "proposal template yang sudah ada dan merangkum isi context sesuai "
        "dengan context proyek yang diberikan. Ikuti DOCUMENT_GENERATOR_WORKFLOW."
    )


class _State(Enum):
    INITIAL = auto()
    RAW_READY = auto()
    PLACEHOLDERS_OBTAINED = auto()
    CONTEXT_SENT = auto()
    DOC_SAVED = auto()


async def run(
    client,
    project_name: str,
    user_query: Optional[str] = None,
    override_template: Optional[str] = None,
    max_turns: int = 12,
    max_parallel_tools: int = 5,
) -> str:
    """Main async workflow untuk pembuatan proposal docx."""

    log = client.logger

    if project_name.lower().endswith((".md", ".txt")):
        project_name = project_name.rsplit(".", 1)[0]

    system_prompt = {"role": "system", "content": _SYSTEM_PROMPT}
    first_user_msg = (
        user_query or f"Buatkan proposal untuk proyek '{project_name}'. Ikuti prosedur."
    )
    messages: List[Dict[str, Any]] = [
        system_prompt,
        {"role": "user", "content": first_user_msg},
    ]

    state: _State = _State.INITIAL
    placeholders: List[str] = []
    doc_path: Optional[str] = None
    retries: Dict[str, int] = {}
    sem = asyncio.Semaphore(max_parallel_tools)

    async def _call_tool(name: str, args: Dict[str, Any]) -> str:
        async with sem:
            log.info(f"Memanggil tool '{name}' arg={args}")
            try:
                raw = await client.call_tool(name, args)
                return raw if isinstance(raw, str) else json.dumps(raw)
            except Exception as e:
                traceback.print_exc()
                return json.dumps({"status": "failure", "error": str(e)})

    def _context_complete(ctx_json: str) -> bool:
        try:
            data = json.loads(ctx_json)
            return isinstance(data, dict) and all(k in data for k in placeholders)
        except Exception:
            return False

    for turn in range(max_turns):
        log.info(f"— Turn {turn + 1}/{max_turns} | state={state.name}")

        explicit_choice: Union[str, Dict[str, Any]] = "auto"
        if state is _State.INITIAL and retries.get("read_project_markdown", 0) == 0:
            explicit_choice = {
                "type": "function",
                "function": {"name": "read_project_markdown"},
            }
        elif (
            state is _State.RAW_READY
            and retries.get("get_template_placeholders", 0) == 0
        ):
            explicit_choice = {
                "type": "function",
                "function": {"name": "get_template_placeholders"},
            }
        elif (
            state in {_State.CONTEXT_SENT, _State.PLACEHOLDERS_OBTAINED}
            and retries.get("generate_proposal_docx", 0) == 0
        ):
            explicit_choice = {
                "type": "function",
                "function": {"name": "generate_proposal_docx"},
            }

        resp = await client.llm.chat.completions.create(
            model=client.model,
            messages=messages,  # type: ignore[arg-type]
            tools=await client.get_tools(),  # type: ignore[arg-type]
            tool_choice=explicit_choice,
        )
        assistant_msg = resp.choices[0].message
        messages.append(assistant_msg.model_dump())

        if not assistant_msg.tool_calls:
            if state is _State.PLACEHOLDERS_OBTAINED:
                ctx_raw = assistant_msg.content or ""
                if not _context_complete(ctx_raw):
                    retries["context"] = retries.get("context", 0) + 1
                    if retries["context"] <= 1:
                        missing = [
                            ph
                            for ph in placeholders
                            if ph not in json.loads(ctx_raw or "{}")
                        ]
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Beberapa placeholder masih kosong: "
                                    f"{missing}. Mohon lengkapi JSON context sepenuhnya."
                                ),
                            }
                        )
                        continue
                    return "Placeholder masih belum lengkap setelah 2× percobaan."

                messages = [system_prompt, {"role": "user", "content": ctx_raw}]
                state = _State.CONTEXT_SENT
                continue

            if state is _State.DOC_SAVED:
                return (
                    assistant_msg.content or f"Proposal berhasil dibuat di {doc_path}"
                )

            continue

        # ----------------------------
        # Tool-calls processing
        # ----------------------------
        tc_results: List[Tuple[str, str, str]] = []
        exec_tasks = []
        tc_extra_args: List[Tuple[str, Dict[str, Any]]] = []
        for tc in assistant_msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            if tc.function.name == "read_project_markdown":
                args.setdefault("project_name", project_name)
            elif tc.function.name == "generate_proposal_docx" and override_template:
                args.setdefault("override_template", override_template)
            exec_tasks.append(_call_tool(tc.function.name, args))
            tc_extra_args.append((tc.function.name, args))

        tool_raw_results = await asyncio.gather(*exec_tasks)

        for tc, raw, (fname, _) in zip(
            assistant_msg.tool_calls, tool_raw_results, tc_extra_args
        ):
            content_str = raw if isinstance(raw, str) else json.dumps(raw)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fname,
                    "content": content_str,
                }
            )
            tc_results.append((fname, content_str, tc.id))

        # ----------------------------
        # Post‑process each tool result
        # ----------------------------
        for fname, content_str, _ in tc_results:
            try:
                payload = json.loads(content_str)
            except json.JSONDecodeError:
                # Fallback untuk string path (generate_proposal_docx)
                payload = (
                    {"status": "success", "path": content_str}
                    if fname == "generate_proposal_docx"
                    else {"status": "raw", "data": content_str}
                )

            # read_project_markdown -------------------------------------
            if fname == "read_project_markdown" and state is _State.INITIAL:
                if payload.get("status") != "success":
                    retries[fname] = retries.get(fname, 0) + 1
                    if retries[fname] <= 1:
                        state = _State.INITIAL
                        break
                    return payload.get("error", "Dokumen proyek tidak ditemukan.")
                messages.append({"role": "user", "content": payload.get("text", "")})
                state = _State.RAW_READY

            # get_template_placeholders ---------------------------------
            elif fname == "get_template_placeholders" and state is _State.RAW_READY:
                placeholders.clear()
                if isinstance(payload, list):
                    placeholders.extend(payload)
                elif isinstance(payload.get("placeholders"), list):
                    placeholders.extend(payload["placeholders"])
                messages.append(
                    {
                        "role": "user",
                        "content": f"Daftar placeholder: {placeholders}",
                    }
                )
                state = _State.PLACEHOLDERS_OBTAINED

            # generate_proposal_docx ------------------------------------
            elif fname == "generate_proposal_docx":
                if payload.get("status") != "success":
                    retries[fname] = retries.get(fname, 0) + 1
                    if retries[fname] <= 1:
                        state = _State.CONTEXT_SENT
                        break
                    return payload.get("error", "Gagal membuat proposal.")
                doc_path = payload.get("path")
                state = _State.DOC_SAVED

        continue

    return doc_path or "Workflow berhenti: mencapai batas maksimum iterasi."
