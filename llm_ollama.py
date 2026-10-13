"""로컬 Ollama 호출. llm_groq.py·llm_nvidia.py와 같은 이름·인터페이스를 갖는다.

Ollama 기본 API(/api/chat)를 이미 있는 httpx로 호출한다. 예외 클래스와 재시도 클라이언트는
llm_nvidia.py의 것을 그대로 가져다 쓴다 (groq SDK와 같은 이름·속성이라 부르는 쪽이 그대로 쓴다).

준비: Ollama 설치 후 `ollama pull qwen2.5:7b`

.env:
    LLM_PROVIDER=ollama
    OLLAMA_MODEL=qwen2.5:7b           # 필수
    OLLAMA_BASE_URL=...               # 선택. 기본 http://localhost:11434
    OLLAMA_NUM_CTX=8192               # 선택. 기본 8192

구조화 출력은 요청의 format에 JSON 스키마를 넣어 Ollama가 생성 단계에서 강제하고, 같은 스키마를
시스템 메시지로도 알려 준다. format만 주면 qwen2.5:7b가 문자열 안에서 같은 글자를 반복하다
max_tokens에서 잘렸다 (2026-10-13 확인).
"""

import logging
import os
import re

import httpx

from llm_nvidia import (  # noqa: F401  (다른 모듈이 llm.APIError 등으로 꺼내 쓴다)
    APIConnectionError,
    APIError,
    APIStatusError,
    BadRequestError,
    NvidiaClient,
    RateLimitError,
    with_schema_prompt,
)

NAME = "ollama"
DEFAULT_BASE_URL = "http://localhost:11434"
# Ollama 기본 컨텍스트는 작아서, 넘치면 프롬프트 앞부분(시스템 메시지·근거 문단)이 조용히 잘린다
DEFAULT_NUM_CTX = 8192
# 로컬 추론은 처음 모델을 올리는 시간까지 더해져 느릴 수 있다
TIMEOUT_SECONDS = 300
THINK_TAG = re.compile(r"<think>.*?</think>", re.DOTALL)

logger = logging.getLogger("llm_ollama")


def get_client(
    max_retries: int | None = None, hooks: list | None = None
) -> NvidiaClient:
    """max_retries는 5xx·연결 실패에서 다시 보내는 횟수, hooks는 응답을 훑어보는 httpx 후크."""
    http = httpx.Client(
        headers={"Accept": "application/json"},
        timeout=TIMEOUT_SECONDS,
        event_hooks={"response": hooks or []},
    )
    base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
    return NvidiaClient(http, base_url, 1 if max_retries is None else max_retries)


def get_model() -> str:
    """사용할 모델 이름. 코드에 기본값을 두지 않고 .env의 OLLAMA_MODEL에서만 읽는다."""
    model = os.getenv("OLLAMA_MODEL", "").strip()
    if not model:
        raise SystemExit(
            '.env에 OLLAMA_MODEL을 설정하세요. 예: OLLAMA_MODEL="qwen2.5:7b"'
        )
    return model


def get_num_ctx() -> int:
    value = os.getenv("OLLAMA_NUM_CTX", "").strip()
    try:
        return int(value) if value else DEFAULT_NUM_CTX
    except ValueError:
        raise SystemExit(
            f".env의 OLLAMA_NUM_CTX가 정수가 아닙니다: {value!r}"
        ) from None


def is_daily_limit(error: Exception) -> bool:
    """로컬 실행이라 하루 한도가 없다."""
    return False


def is_schema_error(error: Exception) -> bool:
    """format으로 생성 단계에서 스키마를 강제하므로 서버가 스키마 오류로 거절하지 않는다."""
    return False


def chat(
    client: NvidiaClient,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    schema: dict | None = None,
    model: str | None = None,
    **extra,
) -> str:
    """LLM 호출은 모두 여기를 거친다. llm_groq.chat과 같은 인자·반환값이다.

    model을 주지 않으면 .env의 OLLAMA_MODEL을 쓴다 (없으면 종료).
    extra는 Ollama options(예: top_p, seed)로 그대로 넘긴다.
    """
    payload = {
        "model": model or get_model(),
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": get_num_ctx(),
            **extra,
        },
    }
    if schema is not None:
        payload["format"] = schema
        payload["messages"] = with_schema_prompt(messages, schema)
    data = client.post("/api/chat", payload)
    try:
        content = data["message"].get("content") or ""
    except (KeyError, TypeError, AttributeError) as e:
        raise APIError(f"응답 형식이 예상과 다름: {str(data)[:200]}", data) from e
    if data.get("done_reason") == "length":
        logger.warning("출력이 max_tokens(%d)에서 잘렸습니다", max_tokens)
    # 추론 모델이 생각 과정을 본문에 섞어 내보내는 경우를 걸러낸다
    return THINK_TAG.sub("", content).strip()
