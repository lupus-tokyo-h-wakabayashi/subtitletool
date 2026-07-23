#!/usr/bin/env python3
import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_MODEL = "qwen3:14b"
STARTUP_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class OllamaGenerateExchange:
    """
    Ollamaとの1回の生成通信結果。

    request_payload:
        /api/generateへ送信したPayload。

    raw_response_body:
        HTTPレスポンスをUTF-8として復号した未加工文字列。

    response_body:
        未加工レスポンスをJSONとして解析した辞書。

    generated_text:
        responseフィールドから取得して前後空白を除去した生成文字列。
    """

    request_payload: dict[str, object]
    raw_response_body: str
    response_body: dict[str, object]
    generated_text: str


def build_generate_payload(
    *,
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    response_format: dict[str, object] | None,
) -> dict[str, object]:
    """
    Ollama /api/generate用のPayloadを生成する。

    response_formatが指定されている場合だけ、
    structured outputs用のformatを追加する。
    """
    payload: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
        },
    }

    if response_format is not None:
        payload["format"] = response_format

    return payload


def get_windows_host_ip() -> str:
    """WSLから見たWindowsホストのIPアドレスを取得する。"""
    result = subprocess.run(
        ["ip", "route"],
        capture_output=True,
        text=True,
        check=True,
    )

    for line in result.stdout.splitlines():
        parts = line.split()

        if line.startswith("default") and "via" in parts:
            return parts[parts.index("via") + 1]

    raise RuntimeError("Windows host IP could not be detected.")


def get_ollama_base_url() -> str:
    return f"http://{get_windows_host_ip()}:11434"


def is_available(timeout: float = 2.0) -> bool:
    """WSLからOllama APIへ接続できるか確認する。"""
    try:
        request = urllib.request.Request(
            f"{get_ollama_base_url()}/api/tags",
            method="GET",
        )

        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status == 200

    except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            OSError,
    ):
        return False


def restart_windows_ollama() -> None:
    """
    Windows側の既存Ollamaを停止し、
    WSLから接続できる0.0.0.0:11434で起動し直す。
    """
    powershell_script = r"""
$ErrorActionPreference = "Stop"

Get-Process ollama -ErrorAction SilentlyContinue |
    Stop-Process -Force -ErrorAction SilentlyContinue

Start-Sleep -Milliseconds 500

$ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"

if (-not (Test-Path $ollama)) {
    throw "ollama.exe not found: $ollama"
}

$env:OLLAMA_HOST = "0.0.0.0:11434"

Start-Process `
    -FilePath $ollama `
    -ArgumentList "serve" `
    -WindowStyle Hidden
"""

    print("Ollama is unavailable. Restarting Windows Ollama...")

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            powershell_script,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(
            f"Failed to restart Windows Ollama: {stderr or 'unknown error'}"
        )


def ensure_available(
    timeout_seconds: int = STARTUP_TIMEOUT_SECONDS,
) -> None:
    """接続不可ならOllamaを起動し直し、利用可能になるまで待機する。"""
    if is_available():
        return

    restart_windows_ollama()

    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        if is_available():
            print(f"Ollama connected: {get_ollama_base_url()}")
            return

        time.sleep(1)

    raise RuntimeError(
        f"Ollama did not become available within {timeout_seconds} seconds: "
        f"{get_ollama_base_url()}"
    )


def generate_exchange(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    top_p: float = 0.9,
    timeout: int = 900,
    response_format: dict[str, object] | None = None,
) -> OllamaGenerateExchange:
    """
    Ollamaへ生成リクエストを送信し、
    実際のRequest Payloadと未加工Responseを含む通信結果を返す。

    response_formatが指定されている場合は、
    JSON Schemaをformatとして送信する。
    """
    ensure_available()

    request_payload = build_generate_payload(
        prompt=prompt,
        model=model,
        temperature=temperature,
        top_p=top_p,
        response_format=response_format,
    )

    payload = json.dumps(
        request_payload
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{get_ollama_base_url()}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            raw_response_body = response.read().decode(
                "utf-8"
            )

    except urllib.error.HTTPError as error:
        detail = error.read().decode(
            "utf-8",
            errors="replace",
        )

        raise RuntimeError(
            f"Ollama HTTP error {error.code}: {detail}"
        ) from error

    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Failed to connect to Ollama: {error.reason}"
        ) from error

    try:
        response_body = json.loads(
            raw_response_body
        )

    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Ollama response is not valid JSON."
        ) from error

    if not isinstance(
        response_body,
        dict,
    ):
        raise RuntimeError(
            "Ollama response body is not a JSON object."
        )

    generated = response_body.get(
        "response"
    )

    if not isinstance(
        generated,
        str,
    ):
        raise RuntimeError(
            "Ollama response does not contain "
            "generated text."
        )

    return OllamaGenerateExchange(
        request_payload=request_payload,
        raw_response_body=raw_response_body,
        response_body=response_body,
        generated_text=generated.strip(),
    )


def generate(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    top_p: float = 0.9,
    timeout: int = 900,
    response_format: dict[str, object] | None = None,
) -> str:
    """
    Ollamaへ生成リクエストを送信し、
    前後空白を除去した生成文字列を返す。

    既存の呼び出し側との互換性を維持するため、
    戻り値は従来どおりstrとする。
    """
    exchange = generate_exchange(
        prompt=prompt,
        model=model,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        response_format=response_format,
    )

    return exchange.generated_text
