#!/usr/bin/env python3
import json
import re
import sys
import urllib.request
from pathlib import Path

MODEL = "qwen3:14b"
CHUNK_SIZE = 40


def get_ollama_url():
    import subprocess
    host = subprocess.check_output(
        "ip route | awk '/default/ {print $3}'",
        shell=True,
        text=True
    ).strip()
    return f"http://{host}:11434/api/generate"


def split_srt_blocks(text):
    return re.split(r"\n\s*\n", text.strip())


def chunk_blocks(blocks, size):
    for i in range(0, len(blocks), size):
        yield blocks[i:i + size]


def build_prompt(srt_chunk):
    return f"""You are a professional subtitle translator.

Translate this SRT from English to natural Japanese.

Rules:
- Preserve subtitle numbers exactly.
- Preserve timestamps exactly.
- Preserve SRT block structure.
- Translate only dialogue text.
- Output ONLY valid SRT.
- Do not add explanations.
- Use natural spoken Japanese.
- Keep names consistent:
  Stargate=スターゲイト
  Destiny=デスティニー
  Colonel Young=ヤング大佐
  Eli=イーライ
  Rush=ラッシュ博士
  Chloe=クロエ
  Scott=スコット
  Lieutenant=中尉
  Senator=上院議員

SRT:
{srt_chunk}
"""


def call_ollama(prompt):
    data = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        get_ollama_url(),
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=600) as res:
        body = json.loads(res.read().decode("utf-8"))
        return body["response"].strip()


def main():
    if len(sys.argv) < 3:
        print("Usage: subtitletool translate input.srt output.srt")
        sys.exit(1)

    input_path = Path(sys.argv[1]).expanduser()
    output_path = Path(sys.argv[2]).expanduser()

    text = input_path.read_text(encoding="utf-8", errors="replace")
    blocks = split_srt_blocks(text)

    translated_chunks = []

    total = len(blocks)
    done = 0

    for chunk in chunk_blocks(blocks, CHUNK_SIZE):
        srt_chunk = "\n\n".join(chunk)
        prompt = build_prompt(srt_chunk)

        print(f"Translating {done + 1}-{done + len(chunk)} / {total} ...")
        translated = call_ollama(prompt)

        translated_chunks.append(translated)
        done += len(chunk)

        output_path.write_text(
            "\n\n".join(translated_chunks).strip() + "\n",
            encoding="utf-8"
        )

    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()
