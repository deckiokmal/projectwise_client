from pathlib import Path
from mcp_client.settings import Settings

cfg = Settings()  # type: ignore


def load_prompt(name: str) -> str:
    """
    Membaca berkas prompt (.txt, UTF-8) dari folder 'prompts'.

    Args:
        name: Nama file prompt—boleh:
              • tanpa ekstensi  -> "kak_analyzer"
              • dengan ekstensi -> "kak_analyzer.md", "kak_analyzer.txt", dsb.
              Fungsi selalu memuat file .txt yang sesuai.

    Returns:
        Isi file prompt sebagai string.

    Raises:
        FileNotFoundError: Jika file tidak ditemukan.
        UnicodeDecodeError: Jika file tidak valid UTF-8.
    """
    # ── Normalisasi nama file ──────────────────────────────────────────────
    raw_name = Path(name).name  # buang path apa pun
    stem = Path(raw_name).stem  # buang ekstensi apa pun
    file_name = f"{stem}.txt"  # pakai ekstensi .txt selalu

    # ── Tentukan direktori prompts ────────────────────────────────────────
    prompt_dir = (
        Path(cfg.prompt_base_path)
        if getattr(cfg, "prompt_base_path", None)
        else Path(__file__).resolve().parent.parent / "prompts"
    )
    prompt_path = prompt_dir / file_name

    # ── Validasi ketersediaan file ────────────────────────────────────────
    if not prompt_path.exists():
        available = sorted(p.name for p in prompt_dir.glob("*.txt"))
        raise FileNotFoundError(
            f"Prompt '{file_name}' tidak ditemukan di {prompt_dir}.\n"
            f"File yang tersedia: {available}"
        )

    # ── Baca & kembalikan isi file ────────────────────────────────────────
    return prompt_path.read_text(encoding="utf-8")
