"""NVIDIA API(OpenAI 호환 chat/completions) 호출. llm_groq.py와 같은 이름·인터페이스를 갖는다.

새 패키지 없이 이미 있는 httpx로 직접 호출한다. groq SDK의 예외와 같은 이름·속성(status_code,
message, body)의 예외를 직접 정의해서, 부르는 쪽(quiz_session, eval_blank)이 그대로 쓴다.

.env:
    LLM_PROVIDER=nvidia
    NVIDIA_API_KEY=...
    NVIDIA_MODEL=...                  # 필수
    NVIDIA_BASE_URL=...               # 선택. 기본 https://integrate.api.nvidia.com/v1
    NVIDIA_STRUCTURED=prompt          # 선택. 출력 형식 강제 방식: prompt | json_schema | guided_json

주의: 아래 세 가지는 NVIDIA 문서로 확인하지 않고 일반적인 OpenAI 호환 방식으로 짠 것이다.
  - 엔드포인트와 Bearer 인증
  - 구조화 출력: 기본 prompt 방식은 스키마를 시스템 메시지에 글로 붙이고 응답 JSON은 부르는 쪽
    코드가 검증한다. 서버가 지원하는 모델이면 json_schema(OpenAI 방식)나 guided_json(NIM의
    nvext)으로 바꿀 수 있다.
  - 하루 한도 판정(is_daily_limit)은 오류 메시지의 일반적인 단어로 추정한다
"""

import json
import logging
import os
import re
import time

import httpx

NAME = "nvidia"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
TIMEOUT_SECONDS = 120
MAX_BACKOFF_SECONDS = 30
THINK_TAG = re.compile(r"<think>.*?</think>", re.DOTALL)

logger = logging.getLogger("llm_nvidia")


# ---------------------------------------------------------------- 예외 (groq SDK와 같은 이름·속성)


class APIError(Exception):
    def __init__(self, message: str, body: object = None):
        super().__init__(message)
        self.message = message
        self.body = body


class APIConnectionError(APIError):
    """서버에 닿지 못함 (네트워크·시간 초과)."""


class APIStatusError(APIError):
    """서버가 오류 상태 코드로 응답함."""

    def __init__(self, message: str, status_code: int, body: object = None):
        super().__init__(message, body)
        self.status_code = status_code


class RateLimitError(APIStatusError):
    """429."""


class BadRequestError(APIStatusError):
    """400."""


# ---------------------------------------------------------------- 클라이언트


class NvidiaClient:
    """httpx 클라이언트와 재시도 설정. 429·5xx는 retry-after(없으면 지수 대기)만큼 기다렸다 다시 보낸다."""

    def __init__(self, http: httpx.Client, base_url: str, max_retries: int):
        self.http = http
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries

    def post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries + 1):
            last = attempt == self.max_retries
            try:
                response = self.http.post(url, json=payload)
            except httpx.TransportError as e:
                if last:
                    raise APIConnectionError(f"연결 실패: {e}") from e
                time.sleep(backoff(attempt, None))
                continue
            if response.status_code < 400:
                return response.json()
            if response.status_code in (429, 500, 502, 503, 504) and not last:
                wait = backoff(attempt, response.headers.get("retry-after"))
                logger.info("HTTP %d, %.1f초 뒤 다시 시도", response.status_code, wait)
                time.sleep(wait)
                continue
            raise error_from(response)
        raise APIConnectionError("재시도 횟수를 넘김")  # 도달하지 않는다


def backoff(attempt: int, retry_after: str | None) -> float:
    try:
        return min(float(retry_after), MAX_BACKOFF_SECONDS)
    except (TypeError, ValueError):
        return min(2.0**attempt, MAX_BACKOFF_SECONDS)


def error_from(response: httpx.Response) -> APIStatusError:
    try:
        body = response.json()
    except ValueError:
        body = response.text
    detail = body.get("error", body) if isinstance(body, dict) else body
    message = detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail)
    text = f"Error code: {response.status_code} - {message}"
    cls = {429: RateLimitError, 400: BadRequestError}.get(response.status_code, APIStatusError)
    return cls(text, response.status_code, body)


def get_client(max_retries: int | None = None, hooks: list | None = None) -> NvidiaClient:
    """max_retries는 429·5xx에서 기다렸다 다시 보내는 횟수, hooks는 응답을 훑어보는 httpx 후크."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit(".env에 NVIDIA_API_KEY가 없습니다")
    http = httpx.Client(
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        timeout=TIMEOUT_SECONDS,
        event_hooks={"response": hooks or []},
    )
    base_url = os.getenv("NVIDIA_BASE_URL", DEFAULT_BASE_URL)
    return NvidiaClient(http, base_url, 2 if max_retries is None else max_retries)


def get_model() -> str:
    """사용할 모델 ID. 코드에 기본값을 두지 않고 .env의 NVIDIA_MODEL에서만 읽는다."""
    model = os.getenv("NVIDIA_MODEL", "").strip()
    if not model:
        raise SystemExit('.env에 NVIDIA_MODEL을 설정하세요. 예: NVIDIA_MODEL="모델 ID"')
    return model


def is_daily_limit(error: Exception) -> bool:
    """하루 한도(토큰·요청·크레딧)에 걸린 429인가. 메시지의 일반적인 단어로 추정한다."""
    text = str(error).lower()
    return any(word in text for word in ("per day", "daily", "quota", "credit"))


def is_schema_error(error: Exception) -> bool:
    """모델 출력이 스키마 검사에 걸려 거절된 400인가. 서버 쪽 강제를 쓰지 않으면 나오지 않는다."""
    return "json_validate_failed" in str(error) or "schema" in str(error).lower()


# ---------------------------------------------------------------- 호출


def with_schema_prompt(messages: list[dict], schema: dict) -> list[dict]:
    """서버가 스키마를 강제하지 않을 때, 스키마를 시스템 메시지로 알려 준다."""
    note = (
        "반드시 아래 JSON 스키마를 따르는 JSON 객체 하나만 출력해. 설명이나 코드 블록은 붙이지 마.\n"
        + json.dumps(schema, ensure_ascii=False)
    )
    return [{"role": "system", "content": note}, *messages]


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

    model을 주지 않으면 .env의 NVIDIA_MODEL을 쓴다 (없으면 종료).
    extra는 모델별 옵션으로 요청 본문에 그대로 넘긴다.
    """
    payload = {
        "model": model or get_model(),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        **extra,
    }
    if schema is not None:
        mode = os.getenv("NVIDIA_STRUCTURED", "prompt").strip().lower()
        if mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "quiz", "schema": schema, "strict": True},
            }
        elif mode == "guided_json":
            payload["nvext"] = {"guided_json": schema}
        else:
            payload["messages"] = with_schema_prompt(messages, schema)
    data = client.post("/chat/completions", payload)
    try:
        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        raise APIError(f"응답 형식이 예상과 다름: {str(data)[:200]}", data) from e
    if choice.get("finish_reason") == "length":
        logger.warning("출력이 max_tokens(%d)에서 잘렸습니다", max_tokens)
    # 추론 모델이 생각 과정을 본문에 섞어 내보내는 경우를 걸러낸다
    return THINK_TAG.sub("", content).strip()
