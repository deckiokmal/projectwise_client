# utils/pipeline_docgen.py
from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any, Dict, List, Optional, Union

from mcp_client.utils.prompt_loader import load_prompt

# ---------------------------------------------------------------------------
# Muat prompt sesuai file document_generator.txt (ekstensi ditangani otomatis)
# ---------------------------------------------------------------------------
try:
    _SYSTEM_PROMPT = load_prompt("document_generator").strip()
except Exception:
    _SYSTEM_PROMPT = (
        'Anda adalah "ProjectWise", asisten virtual untuk tim Presales & '
        "Project Manager. Tugas Anda adalah generate document docx berdasarkan "
        "proposal template yang sudah ada dan merangkum isi context sesuai "
        "dengan context proyek yang diberikan. Ikuti DOCUMENT_GENERATOR_WORKFLOW."
    )


# ---------------------------------------------------------------------------
class _State:
    INITIAL = "INITIAL"
    RAW_READY = "RAW_READY"
    PLACEHOLDERS_OBTAINED = "PLACEHOLDERS_OBTAINED"
    CONTEXT_SENT = "CONTEXT_SENT"
    DOC_SAVED = "DOC_SAVED"


# ---------------------------------------------------------------------------
async def run(
    client,
    project_name: str,
    user_query: Optional[str] = None,
    override_template: Optional[str] = None,
    max_turns: int = 12,
    max_parallel_tools: int = 5,
) -> str:
    log = client.logger

    # Normalize project_name tanpa .md/.txt
    if project_name.lower().endswith(".md") or project_name.lower().endswith(".txt"):
        project_name = project_name.rsplit(".", 1)[0]

    # Inisialisasi chat
    system_prompt = {"role": "system", "content": _SYSTEM_PROMPT}
    user_msg = (
        user_query or f"Buatkan proposal untuk proyek '{project_name}'. Ikuti prosedur."
    )
    messages: List[Dict[str, Any]] = [
        system_prompt,
        {"role": "user", "content": user_msg},
    ]

    state: str = _State.INITIAL
    raw_context: Optional[str] = None
    placeholders: List[str] = []
    doc_path: Optional[str] = None

    # Retry counter per tool
    retry_tracker: Dict[str, int] = {}

    sem = asyncio.Semaphore(max_parallel_tools)

    async def _call_tool(name: str, args: Dict[str, Any]) -> Union[str, Any]:
        async with sem:
            log.info(f"Memanggil tool '{name}' arg={args}")
            try:
                return await client.call_tool(name, args)
            except Exception as e:
                traceback.print_exc()
                return json.dumps({"status": "failure", "error": str(e)})

    def _context_complete(raw: str) -> bool:
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                return False
            return all(ph in data for ph in placeholders)
        except Exception:
            return False

    # Main loop chat
    for turn in range(max_turns):
        log.info(f"— Turn {turn + 1}/{max_turns} | state={state}")

        # Tentukan tool_choice
        if (
            state == _State.INITIAL
            and retry_tracker.get("read_project_markdown", 0) == 0
        ):
            tool_choice: Any = {
                "type": "function",
                "function": {"name": "read_project_markdown"},
            }
        elif (
            state == _State.RAW_READY
            and retry_tracker.get("get_template_placeholders", 0) == 0
        ):
            tool_choice = {
                "type": "function",
                "function": {"name": "get_template_placeholders"},
            }
        elif (
            state == _State.CONTEXT_SENT
            and retry_tracker.get("generate_proposal_docx", 0) == 0
        ):
            tool_choice = {
                "type": "function",
                "function": {"name": "generate_proposal_docx"},
            }
        else:
            tool_choice = "auto"

        resp = await client.llm.chat.completions.create(
            model=client.model,
            messages=messages,  # type: ignore[arg-type]
            tools=await client.get_tools(),  # type: ignore[arg-type]
            tool_choice=tool_choice,
        )
        assistant_msg = resp.choices[0].message
        messages.append(assistant_msg.model_dump())

        # Tanpa tool-call
        if not assistant_msg.tool_calls:
            if state == _State.PLACEHOLDERS_OBTAINED:
                # LLM mengirim context JSON
                ctx_raw = assistant_msg.content or ""
                if not _context_complete(ctx_raw):
                    retry_tracker.setdefault("context", 0)
                    if retry_tracker["context"] == 0:
                        retry_tracker["context"] = 1
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Beberapa placeholder masih kosong: "
                                    f"{[ph for ph in placeholders if ph not in json.loads(ctx_raw or '{}')]}. "
                                    "Mohon lengkapi dalam format JSON konteksnya."
                                ),
                            }
                        )
                        # Tetap di state PLACEHOLDERS_OBTAINED
                        continue
                    client.logger.warning(
                        "Placeholder masih belum lengkap setelah 2× percobaan. Workflow dihentikan."
                    )
                # context lengkap → reset chat untuk render
                messages = [system_prompt, {"role": "user", "content": ctx_raw}]
                state = _State.CONTEXT_SENT
                continue

            if state == _State.DOC_SAVED:
                return assistant_msg.content or "Proposal berhasil dibuat."

            # Message biasa, lanjut
            continue

        # Ada tool-call → eksekusi paralel
        tc_meta: List[tuple[str, str]] = []
        exec_tasks = []
        for tc in assistant_msg.tool_calls:
            try:
                tc_args = json.loads(tc.function.arguments or "{}")
            except Exception:
                tc_args = {}

            if tc.function.name == "read_project_markdown":
                tc_args.setdefault("project_name", project_name)
            elif tc.function.name == "generate_proposal_docx" and override_template:
                tc_args.setdefault("override_template", override_template)

            exec_tasks.append(_call_tool(tc.function.name, tc_args))
            tc_meta.append((tc.id, tc.function.name))

        results = await asyncio.gather(*exec_tasks, return_exceptions=True)

        # Proses hasil setiap tool-call
        for (tc_id, fname), res in zip(tc_meta, results):
            content = res if isinstance(res, str) else res
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": fname,
                    "content": content,
                }
            )

            # Parse payload JSON bila mungkin
            try:
                payload = json.loads(content)  # type: ignore
            except Exception:
                payload = {}

            # STEP-1: read_project_markdown
            if fname == "read_project_markdown":
                if payload.get("status") != "success":
                    retry_tracker[fname] = retry_tracker.get(fname, 0) + 1
                    if retry_tracker[fname] <= 1:
                        state = _State.INITIAL  # beri LLM retry
                        break
                    return payload.get("error", "Dokumen proyek tidak ditemukan.")
                # Success → kirim raw_context
                raw_context = payload.get("text", "")
                messages.append({"role": "user", "content": raw_context})
                state = _State.RAW_READY
                continue

            # STEP-2: get_template_placeholders
            if fname == "get_template_placeholders":
                if isinstance(payload, list):
                    placeholders = payload
                else:
                    placeholders = []
                # Kirim daftar placeholder ke LLM
                messages.append(
                    {"role": "user", "content": f"Daftar placeholder: {placeholders}"}
                )
                state = _State.PLACEHOLDERS_OBTAINED
                continue

            # STEP-3: generate_proposal_docx
            if fname == "generate_proposal_docx":
                if payload.get("status") != "success":
                    retry_tracker[fname] = retry_tracker.get(fname, 0) + 1
                    if retry_tracker[fname] <= 1:
                        state = _State.CONTEXT_SENT  # beri LLM retry
                        break
                    return payload.get("error", "Gagal membuat proposal.")
                doc_path = payload.get("path")
                state = _State.DOC_SAVED
                continue

        else:
            # loop selesai tanpa break, lanjut ke next turn
            continue

    return doc_path or "Workflow berhenti: mencapai batas maksimum iterasi."
