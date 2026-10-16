# AGENTS.md

이 저장소에서 일하는 코딩 에이전트를 위한 지침이에요. 사용법·명령·구조는 `README.md`가 기준이에요.

## 프로젝트

`check_your_learning_by_AI`는 학습자의 질문에 교재 문단을 근거로 답하고, 답변에 쓴 바로 그 문단에서 확인 퀴즈를 내는 AI 도구예요. 메타인지 점검과 장기 기억 유지가 목표예요.

## 응답 규칙

- 한국어로 간결하게, "~해요" 체로 써요.

## 기술 스택

- Python 3.13, conda(환경 이름 `check_your_learning`, 채널 conda-forge), Ruff, 로컬 실행
- 패키지는 사용자가 직접 설치해요. 필요하면 `conda install -c conda-forge ...` 명령만 제안해요.

## 코드 규칙

- 린트·포맷은 Ruff로 통일해요 (`ruff check`, `ruff format`).
- 예외는 구체적인 타입으로 잡아요 (bare `except:` 금지).
- 환경변수는 python-dotenv로 읽어요. `.env`는 커밋하지 않아요.
- 파일을 잘게 쪼개지 말고 지금 파일 개수 내외로 유지해요.

## 실행할 때 주의

- 명령은 프로젝트 루트에서, `evals/`·`demo/`·`tools/`는 `python -m`으로 실행해요.
- conda python을 경로로 직접 부르면 matplotlib·sklearn에서 0xC06D007F로 조용히 죽어요. PowerShell에서는 먼저 PATH를 활성화 상태처럼 맞춰요:
  `$e='C:\Users\302\anaconda3\envs\check_your_learning'; $env:PATH = "$e;$e\Library\mingw-w64\bin;$e\Library\usr\bin;$e\Library\bin;$e\Scripts;$e\bin;" + $env:PATH`
- Streamlit 데모는 실행 명령에서 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`을 먼저 설정해요 (README "시연 GUI").
- LLM 호출은 하루 토큰 한도가 있어요. 평가(`evals.eval_blank`)를 크게 돌리기 전에 범위를 확인받아요.

## 절대 금지

- 삭제 명령(`rm`, `del`, `rmdir`, `git clean` 등)은 쓰지 않아요. 없앨 파일은 `archive/`로 옮겨요.
  - 경로: `archive/{종류}/{날짜}/{파일명_날짜}` (예: `archive/quiz_session/20261008/quiz_session_20261008.py`)
- 시스템 설정 파일은 수정하지 않아요.
- 외부 네트워크 요청(위키독스 수집 `--crawl`, 모델 다운로드, 웹 검색 등) 전에는 반드시 확인받아요. LLM API 호출은 사용자가 요청한 실행 범위 안에서만 해요.

## Git 규칙

- 작업은 작업 브랜치에서 해요. 브랜치 이름은 `.scratch/<feature-slug>/`의 slug를 그대로 써요 (티켓별 브랜치가 필요하면 `<slug>-NN`).
- 에이전트는 작업 브랜치에 커밋까지 해요. `main` 병합과 `push`는 사용자가 해요.
- 커밋 메시지는 기존 형식 `범위:설명`을 따라요 (예: `quiz_templates:빈칸개수검증`). 범위는 모듈·영역 이름이에요. 티켓 작업이면 본문에 티켓 경로를 적어요.
- 스테이징은 바꾼 파일을 경로로 지정해요. `.data/` 산출물은 사용자가 필요할 때 직접 올려요.
- 기록을 지우는 명령(`reset --hard`, `branch -D`, `push --force`, rebase로 커밋 버리기)은 쓰지 않아요.
- 작업을 마치면 브랜치 이름과 커밋 목록을 알려 줘요.

## 기록 위치

- 이슈·명세: `.scratch/` (아래 Issue tracker)
- 용어: `GLOSSARY.md`, 되돌리기 어려운 결정: `docs/adr/`
- 기획·실험 기록(커밋 제외): `.idea_folder/`. 설계 문서 `정확도_2차_근거문장_인덱스_설계.md` 맨 위 "최종 결정"이 최신이고, 아래 장들은 옛 방식의 기록이에요.

## Agent skills

### Issue tracker

이슈와 명세는 `.scratch/<feature-slug>/` 아래 로컬 Markdown 파일이에요. See `docs/agents/issue-tracker.md`.

### Triage labels

기본 라벨 5개(`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`)를 이슈 파일의 `Status:` 줄에 써요. See `docs/agents/triage-labels.md`.

### Domain docs

single-context: 루트 `GLOSSARY.md` + `docs/adr/`. See `docs/agents/domain.md`.
