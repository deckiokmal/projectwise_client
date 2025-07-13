# utils/pipeline_kak.py  (versi perbaikan)
from __future__ import annotations
import asyncio
import json
import traceback
from typing import List, Dict, Any
from mcp_client.utils.prompt_loader import load_prompt


async def run(
    client,
    query: str,
    prompt_instruction_name: str,
    kak_tor_md_name: str,
    max_turns: int = 10,
    max_parallel_tools: int = 5,
) -> str:
    log = client.logger
    system_prompt = {
        "role": "system",
        "content": load_prompt("kak_analyzer").strip(),
    }
    messages: List[Dict[str, Any]] = [
        system_prompt,
        {"role": "user", "content": query},
    ]

    state = "INITIAL"
    summary_json: str | None = None
    original_history: List[Dict[str, Any]] = []
    sem = asyncio.Semaphore(max_parallel_tools)

    def _payload_success(raw: str) -> bool:
        try:
            data = json.loads(raw)
            return (
                isinstance(data, dict)
                and isinstance(data.get("instruction"), str)
                and isinstance(data.get("context"), str)
            )
        except Exception:
            return False

    # -------------------------------------------------- #
    async def _exec_tool(tc):
        async with sem:
            fname = tc.function.name
            try:
                fargs = json.loads(tc.function.arguments or "{}")
            except Exception:
                fargs = {}

            if fname == "build_summary_tender_payload":
                fargs.setdefault("prompt_instruction_name", prompt_instruction_name)
                fargs.setdefault("kak_tor_md_name", kak_tor_md_name)

            log.info(f"Memanggil tool '{fname}' arg={fargs}")
            try:
                result = await client.call_tool(fname, fargs)
            except Exception as e:
                log.error(f"Error tool '{fname}': {e}")
                traceback.print_exc()
                result = f"Error executing tool {fname}: {e}"

            return {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": fname,
                "content": result,
                "fname": fname,
                "fargs": fargs,
            }

    # -------------------------------------------------- #

    for turn in range(max_turns):
        log.info(f"— Turn {turn + 1}/{max_turns} | state={state}")

        tools = await client.get_tools()
        tool_mode = "none" if state == "PAYLOAD_SENT" else "auto"
        resp = await client.llm.chat.completions.create(
            model=client.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_mode,
        )
        assistant_message = resp.choices[0].message
        messages.append(assistant_message.model_dump())

        # ── 1. Tidak ada tool-call
        if not assistant_message.tool_calls:
            if state == "PAYLOAD_SENT":
                summary_json = assistant_message.content or ""
                if not summary_json.strip():
                    return "Ringkasan kosong."
                # try:
                #     json.loads(summary_json)
                # except json.JSONDecodeError:
                #     return "Ringkasan bukan JSON valid."

                messages = (
                    [system_prompt]
                    + original_history
                    + [assistant_message.model_dump()]
                )
                state = "SUMMARY_OBTAINED"
                continue

            if state == "SAVED":
                return summary_json or (assistant_message.content or "")

            return assistant_message.content or "Tidak ada jawaban."

        # ── 2. Ada tool-call → eksekusi paralel
        tasks = [_exec_tool(tc) for tc in assistant_message.tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        reset_after_payload = False
        for tc, res in zip(assistant_message.tool_calls, results):
            if isinstance(res, Exception):
                log.error(f"Tool {tc.function.name} gagal: {res}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": str(res),
                        "error": True,
                    }
                )
                continue

            fname = res.pop("fname")  # type: ignore[arg-type]
            fargs = res.pop("fargs")  # type: ignore[arg-type]
            content = res["content"]  # type: ignore
            messages.append(res)  # type: ignore

            if fname == "build_summary_tender_payload":
                if _payload_success(content):
                    # Berhasil ambil payload
                    original_history = [
                        {"role": "user", "content": query},
                        assistant_message.model_dump(),
                        res,  # type: ignore
                    ]
                    messages = [system_prompt, {"role": "user", "content": content}]
                    state = "PAYLOAD_SENT"
                    reset_after_payload = True
                else:
                    # Error file not found → tetap di INITIAL; LLM bisa coba lagi
                    state = "INITIAL"

            elif fname == "save_summary_markdown_tool":
                if summary_json is None:
                    summary_json = fargs.get("summary", "")
                state = "SAVED"

        if reset_after_payload:
            continue

    log.warning("Mencapai batas maksimum turn tanpa jawaban final.")
    return summary_json or "Proses mencapai batas maksimum turn."
