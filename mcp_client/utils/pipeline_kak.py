# utils/pipeline_kak.py
from __future__ import annotations
import asyncio
import json
import traceback
from typing import List, Dict, Any
from mcp_client.utils.prompt_loader import load_prompt

# ---------------------------------------------------------------------------
#  Prompt system – di‑load dari folder prompts/ atau hard‑coded sebagai fallback
# ---------------------------------------------------------------------------
try:
    _SYSTEM_PROMPT = load_prompt("kak_analyzer").strip()
except (FileNotFoundError, IOError):
    # Fallback ke prompt bawaan jika file belum tersedia
    _SYSTEM_PROMPT = (
        "Anda adalah “ProjectWise”, asisten virtual untuk tim Presales & Project Manager."
        "Tugas Anda adalah menganalisis dokumen KAK/TOR tender dan merangkum poin-poin"
        "penting ke dalam format JSON terstruktur. Ikuti "
        "prosedur di bawah TANPA"
        "menyimpang, dan gunakan SELALU bahasa Indonesia."
    )


# ---------------------------------------------------------------------------
#  Status enum simpel
# ---------------------------------------------------------------------------
class _State:
    INITIAL = "INITIAL"  # belum memanggil tool apa pun
    PAYLOAD_SENT = "PAYLOAD_SENT"  # LLM sudah mengirim payload
    SUMMARY_OBTAINED = "SUMMARY_OBTAINED"  # Ringkasan sudah diterima
    SAVED = "SAVED"


# ---------------------------------------------------------------------------
#  Pipeline utama
# ---------------------------------------------------------------------------
async def run(
    client,
    user_query: str,
    prompt_instruction_name: str,
    kak_tor_md_name: str,
    max_turns: int = 10,
    max_parallel_tools: int = 5,
) -> str:
    log = client.logger
    system_prompt = {
        "role": "system",
        "content": _SYSTEM_PROMPT,
    }
    messages: List[Dict[str, Any]] = [
        system_prompt,
        {"role": "user", "content": user_query},
    ]

    state = _State.INITIAL
    summary_json: str | None = None
    original_history: List[Dict[str, Any]] = []
    sem = asyncio.Semaphore(max_parallel_tools)

    # -------------------------------------------------------#
    #  Helper: validasi apakah payload hasil tool sukses
    # -------------------------------------------------------#
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

    # -------------------------------------------------------#
    #  Helper: eksekusi sebuah tool‑call (dipanggil paralel)
    # -------------------------------------------------------#
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

    # -------------------------------------------------------#
    #  Main chat loop
    # -------------------------------------------------------#
    for turn in range(max_turns):
        log.info(f"— Turn {turn + 1}/{max_turns} | state={state}")

        tools = await client.get_tools()
        tool_mode = "none" if state == _State.PAYLOAD_SENT else "auto"
        resp = await client.llm.chat.completions.create(
            model=client.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_mode,
        )
        assistant_message = resp.choices[0].message
        messages.append(assistant_message.model_dump())

        # ─────────────────────────────────────────── #
        #  1. Tidak ada tool‑call pada balasan LLM
        # ─────────────────────────────────────────── #
        if not assistant_message.tool_calls:
            if state == _State.PAYLOAD_SENT:
                summary_json = assistant_message.content or ""
                if not summary_json.strip():
                    return "Ringkasan kosong."

                # Reset pesan: hanya system + original history + ringkasan
                messages = (
                    [system_prompt]
                    + original_history
                    + [assistant_message.model_dump()]
                )
                state = _State.SUMMARY_OBTAINED
                continue

            if state == _State.SAVED:
                # Workflow selesai, kembalikan hasil
                return summary_json or (assistant_message.content or "")

            # Belum ada yang bisa diproses
            return assistant_message.content or "Tidak ada jawaban."

        # ─────────────────────────────────────────── #
        #  2. Ada tool-call → eksekusi paralel
        # ─────────────────────────────────────────── #
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
                        {"role": "user", "content": user_query},
                        assistant_message.model_dump(),
                        res,  # type: ignore
                    ]
                    messages = [system_prompt, {"role": "user", "content": content}]
                    state = _State.PAYLOAD_SENT
                    reset_after_payload = True
                else:
                    # Error file not found → tetap di INITIAL; LLM bisa coba lagi
                    state = _State.INITIAL

            elif fname == "save_summary_markdown_tool":
                if summary_json is None:
                    summary_json = fargs.get("summary", "")
                state = _State.SAVED

        if reset_after_payload:
            continue

    log.warning("Mencapai batas maksimum turn tanpa jawaban final.")
    return summary_json or "Proses mencapai batas maksimum turn."
