"""정확도 테스트 공용: 케이스 CSV 읽기, Groq 모델 옵션, 하루 한도 예외.

1차 O/X 테스트(eval_accuracy.py, archive로 이동)에서 빼 왔다. eval_blank.py가 쓴다.
"""

import csv
from pathlib import Path

from text_ingestion import DATA_ROOT

PROJECT_ROOT = Path(__file__).resolve().parent
CASES_CSV = (
    PROJECT_ROOT / ".idea_folder" / "testcase" / "테스트 케이스.정확도 - 시트1.csv"
)
OUT_DIR = DATA_ROOT / "eval"
# 모델은 .env의 GROQ_MODEL로 정한다 (quiz_session.get_model).
# 모델이 바뀌면 이전 결과와 직접 비교할 수 없으니 결과 파일 이름에 모델명을 넣는다
# 추론 모델은 생각 토큰도 max_tokens에 포함돼 작으면 JSON 전에 잘린다.
# 추론 강도를 낮추고 상한을 넉넉히 잡는다 (Groq 추론 모델 옵션)
MODEL_OPTIONS = {
    "openai/gpt-oss": {"max_tokens": 800, "reasoning_effort": "low"},
}
DEFAULT_MAX_TOKENS = 200  # 분당 출력 토큰 한도(1,000)를 아끼려고 작게 잡는다
TEMPERATURE = 0.3
GROQ_MAX_RETRIES = 6  # 분당 한도에 걸리면 SDK가 대기 후 재시도한다
MAX_API_ERRORS = 5  # API 오류가 연속으로 이만큼 쌓이면 실행을 멈춘다


class DailyLimitReached(Exception):
    """Groq 하루 한도(토큰·요청). 기다려도 바로 풀리지 않으니 실행을 멈춘다."""


class ApiErrorsPiledUp(Exception):
    """API 오류가 연속으로 쌓였다. 모델 ID·키·서버 문제일 가능성이 커서 실행을 멈춘다."""


def model_options(model: str) -> dict:
    for prefix, options in MODEL_OPTIONS.items():
        if model.startswith(prefix):
            return dict(options)
    return {"max_tokens": DEFAULT_MAX_TOKENS}


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("id", "").strip()]


def titles_by_book_id(index: dict[str, dict]) -> dict[int, str]:
    """chunks의 book_id → book_title. 케이스 CSV는 book_id만 갖고, 프롬프트에 쓸 책 이름은 여기서 찾는다."""
    titles = {}
    for chunk in index.values():
        if chunk.get("book_title") and chunk.get("book_id") is not None:
            titles[chunk["book_id"]] = chunk["book_title"]
    return titles


def parse_book_id(value: str) -> int | None:
    try:
        return int(value.strip())
    except ValueError:
        return None
