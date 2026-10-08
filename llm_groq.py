"""Groq API 호출. llm_nvidia.py와 같은 이름·인터페이스를 갖는다.

quiz_session.py가 .env의 LLM_PROVIDER(groq 또는 nvidia)에 따라 둘 중 하나를 불러 쓴다.
공통 인터페이스: NAME, get_client(), get_model(), chat(), is_daily_limit(), is_schema_error(),
그리고 예외 클래스 APIError, APIConnectionError, APIStatusError, RateLimitError, BadRequestError.

.env:
    LLM_PROVIDER=groq
    GROQ_API_KEY=...
    GROQ_MODEL=...   # 필수
"""

import logging
import os
import re

from groq import (  # noqa: F401  (다른 모듈이 llm.APIError 등으로 꺼내 쓴다)
    APIConnectionError,
    APIError,
    APIStatusError,
    BadRequestError,
    Groq,
    RateLimitError,
)

NAME = "groq"
THINK_TAG = re.compile(r"<think>.*?</think>", re.DOTALL)

logger = logging.getLogger("llm_groq")


def get_client(max_retries: int | None = None, hooks: list | None = None):
    """max_retries는 429 등에서 SDK가 기다렸다 다시 보내는 횟수, hooks는 응답을 훑어보는 httpx 후크."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(".env에 GROQ_API_KEY가 없습니다")
    options: dict = {}
    if max_retries is not None:
        options["max_retries"] = max_retries
    if hooks:
        import httpx

        options["http_client"] = httpx.Client(event_hooks={"response": hooks})
    return Groq(api_key=api_key, **options)


def get_model() -> str:
    """사용할 모델 ID. 코드에 기본값을 두지 않고 .env의 GROQ_MODEL에서만 읽는다."""
    model = os.getenv("GROQ_MODEL", "").strip()
    if not model:
        raise SystemExit('.env에 GROQ_MODEL을 설정하세요. 예: GROQ_MODEL="모델 ID"')
    return model


def is_daily_limit(error: Exception) -> bool:
    """하루 한도(토큰·요청)에 걸린 429인가. 기다려도 바로 풀리지 않는다."""
    return "per day" in str(error)


def is_schema_error(error: Exception) -> bool:
    """모델 출력이 스키마 검사에 걸려 거절된 400인가 (같은 입력으로 다시 시도할 만하다)."""
    return "json_validate_failed" in str(error)


def chat(
    client,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    schema: dict | None = None,
    model: str | None = None,
    **extra,
) -> str:
    """LLM 호출은 모두 여기를 거친다.

    model을 주지 않으면 .env의 GROQ_MODEL을 쓴다 (없으면 종료).
    extra는 모델별 옵션(예: 추론 모델의 reasoning_effort)으로 그대로 넘긴다.
    """
    options = dict(extra)
    if schema is not None:
        # strict가 아니면 Groq는 생성을 제한하지 않고 생성 후 검사만 해서,
        # 모델이 스키마를 무시하면 400(json_validate_failed)으로 통째로 거절된다
        options["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "quiz", "schema": schema, "strict": True},
        }
    response = client.chat.completions.create(
        model=model or get_model(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **options,
    )
    choice = response.choices[0]
    if choice.finish_reason == "length":
        logger.warning("출력이 max_tokens(%d)에서 잘렸습니다", max_tokens)
    # 추론 모델이 생각 과정을 본문에 섞어 내보내는 경우를 걸러낸다
    return THINK_TAG.sub("", choice.message.content or "").strip()
