# LLM 공급자는 같은 인터페이스의 파일로 분리하고 .env로 고른다

`llm_groq.py`, `llm_nvidia.py`, `llm_ollama.py`가 같은 함수(`get_client`, `get_model`, `chat`, `is_daily_limit`, `is_schema_error`)와 예외를 갖고, `quiz_session.py`가 `.env`의 `LLM_PROVIDER`로 골라요. NVIDIA는 새 패키지 없이 `httpx`로 직접 호출하고, 구조화 출력은 `NVIDIA_STRUCTURED=prompt`만 안정적이었어요.

## Consequences

- 데모 기준 모델은 Groq `qwen/qwen3.8-27b`예요. 이 PC(GTX 950 2GB)에서 Ollama는 CPU로 돌아 질문당 2~4분이라 오프라인 시연용으로만 써요 (`.idea_folder/데모_LLM_모델비교_20261013.md`).
- 분당 토큰 한도(429)는 SDK 재시도에 맡기고 따로 대응하지 않아요.
