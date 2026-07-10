#!/usr/bin/env python3
import json
import subprocess
import urllib.request


def get_ollama_base_url() -> str:
    host = subprocess.check_output(
        "ip route | awk '/default/ {print $3}'",
        shell=True,
        text=True,
    ).strip()

    return f"http://{host}:11434"


def generate(
    prompt: str,
    model: str = "qwen3:14b",
    temperature: float = 0.2,
    top_p: float = 0.9,
    timeout: int = 900,
) -> str:
    data = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{get_ollama_base_url()}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as res:
        body = json.loads(res.read().decode("utf-8"))
        return body["response"].strip()