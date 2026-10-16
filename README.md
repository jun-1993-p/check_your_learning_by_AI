# check_your_learning_by_AI

학습자의 메타인지를 점검하고, 학습 직후 확인 퀴즈를 즉시 제공해 장기 기억 유지 효과를 높이는 AI 도구입니다.

학습자가 질문하면 교재 본문에서 근거 문단을 검색해 답변하고, **답변에 사용된 바로 그 문단**에서 퀴즈를 출제합니다. 모든 문제는 근거 문단과 출처 URL을 함께 제시하며, 문제와 정답이 실제로 문단에 존재하는지 코드가 한 번 더 검증합니다.

## 주요 특징

- **근거 기반 답변(RAG)**: 교재 본문 문단만을 근거로 답변하며, 문장마다 인용 번호(`[1]`, `[2]`)를 표기합니다. 자료에 답이 없으면 추측하지 않습니다.
- **문단 단위 인덱스**: 설명 블록과 코드·표를 묶은 200~600자 내외의 문단(A+ 문단)을 검색 단위로 사용합니다.
- **4가지 퀴즈 유형**: O/X, 빈칸, 4지선다, 단답형. 유형은 LLM이 문단 성격에 맞춰 문단별로 선택합니다.
- **출제 검증**: LLM이 제시한 근거 인용(`evidence`)과 정답이 문단 원문에 있는지, 보기 중복·정답 번호 범위·정답 노출 여부 등을 코드로 재검사하고, 실패하면 다른 문단으로 재시도합니다.
- **메타인지 점검**: CLI 풀이 모드(`--solve`)에서 정답 확인 전 확신도(모름/애매/확실)를 먼저 입력합니다. 시연 GUI는 확신도 입력 없이 바로 채점합니다.
- **LLM 공급자 교체**: `.env`의 `LLM_PROVIDER` 값만 바꿔 Groq와 NVIDIA를 전환합니다.
- **증분 처리**: 내용 해시를 기준으로 변경된 파일·벡터만 갱신합니다.

## 동작 흐름

```
[위키독스] ──수집──▶ raw HTML ──정제──▶ chunks.jsonl ──문단화·임베딩──▶ Chroma(bge-m3)
 text_ingestion.py      text_refine.py       text_paragraphs.py
                                                      │
 학습자 질문 ──키워드 검색(거리 필터)──▶ 근거 문단 ──▶ 답변 + 인용
                                                      │
                          인용 문단 + 같은 페이지의 다른 문단 ──▶ 문제 출제 ──▶ 검증 ──▶ 풀이·채점
                                        quiz_session.py / quiz_templates.py
```

1. **수집**: 위키독스 책의 목차와 페이지 HTML을 `.data/book{N}/raw/`에 캐시합니다.
2. **정제**: HTML을 블록(문단, 코드, 표, 목록 등) 단위로 구조화해 `chunks.jsonl`(페이지 1개 = 1행)로 저장합니다. 안내·목차·부속 페이지는 자동 제외하며, 코드 블록은 `code_blocks.jsonl`에 별도 저장합니다.
3. **문단화·임베딩**: `chunks.jsonl`에서 A+ 문단을 생성하고, `BAAI/bge-m3`로 임베딩해 Chroma에 저장합니다.
4. **검색·답변**: 질문에서 키워드만 남겨(의문사·서술어 제거, 조사 떼기. 예: "튜플의 append 메서드는 어떻게 써?" → "튜플 append 메서드") 검색하고, 코사인 거리가 가까운 문단만(`MAX_DISTANCE=0.45`, 1위 대비 `0.15` 이내) 근거로 채택합니다. 답변에는 원래 질문을 씁니다. 무관한 질문은 LLM 호출 없이 종료됩니다.
5. **출제**: 답변이 인용한 문단과 같은 페이지의 다른 문단을 약 2:1로 섞어 문제를 만듭니다.

## 요구 사항

| 항목 | 내용 |
| --- | --- |
| 언어/런타임 | Python 3.13 |
| 패키지 관리 | conda |
| 린트·포맷 | Ruff |
| 실행 환경 | 로컬 |

| 용도 | 패키지 |
| --- | --- |
| 기본 (수집·임베딩·답변·퀴즈·평가) | `python-dotenv`, `beautifulsoup4`, `numpy`, `sentence-transformers`, `chromadb`, `httpx`, `groq` |
| 시연 GUI (`demo/demo_app.py`) | `streamlit` |
| 시각화 (`tools/text_visualize.py`, 선택) | `matplotlib`, `pandas`, `scikit-learn`, `plotly`, `umap-learn` |
| 개발 도구 | `ruff` |

- `sentence-transformers`를 conda-forge에서 설치하면 PyTorch(`pytorch`)와 `transformers`가 함께 설치됩니다. 임베딩은 기본적으로 CPU에서 실행합니다(`EMBED_DEVICE=cpu`).
- `torchvision`은 필요하지 않습니다. 데모 실행 중 `No module named 'torchvision'` 로그는 Streamlit 파일 감시기 때문이며, `.streamlit/config.toml`에서 꺼 두었습니다.
- NVIDIA 공급자(`llm_nvidia.py`)는 `httpx`로 호출하므로 `openai` 패키지가 필요 없습니다.

> 위 패키지와 검증한 버전은 `environment.yml`에 정리되어 있습니다. pip용 `requirements.txt`는 두지 않습니다.

### 검증한 버전

아래 버전 조합에서 동작을 확인했습니다 (2026-10-12, Windows 11, conda-forge).

| 패키지 | 버전 | 패키지 | 버전 |
| --- | --- | --- | --- |
| python | 3.13.15 | streamlit | 1.65.0 |
| sentence-transformers | 6.1.0 | ruff | 0.16.8 |
| torch (pytorch) | 2.10.0 | matplotlib | 3.11.2 |
| transformers | 5.17.0 | pandas | 3.0.6 |
| chromadb | 1.5.9 | scikit-learn | 1.9.1 |
| protobuf | 6.33.5 | plotly | 7.1.0 |
| numpy | 2.5.3 | umap-learn | 0.5.12 |
| httpx | 0.28.1 | groq | 1.7.0 |
| beautifulsoup4 | 4.15.0 | python-dotenv | 1.2.2 |

> chromadb(conda-forge)의 opentelemetry-proto는 protobuf를 순수 Python 구현으로 써야 import됩니다. CLI는 코드가 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`을 설정하고, 데모는 실행 명령에서 설정합니다([시연 GUI](#3-시연-gui-streamlit) 참고).

## 설치 및 설정

`environment.yml`로 한 번에 만듭니다 (검증한 버전 그대로, 시연·시각화·개발 도구 포함).

```bash
conda env create -f environment.yml
conda activate check_your_learning
```

패키지를 용도별로 직접 설치하려면 아래처럼 합니다.

```bash
# 1) conda 환경 생성
conda create -n check_your_learning python=3.13
conda activate check_your_learning

# 2) 기본 패키지
conda install -c conda-forge python-dotenv beautifulsoup4 numpy sentence-transformers chromadb httpx groq

# 3) 시연 GUI
conda install -c conda-forge streamlit

# 4) 시각화 (선택)
conda install -c conda-forge matplotlib pandas scikit-learn plotly umap-learn

# 5) 개발 도구
conda install -c conda-forge ruff
```

명령은 항상 `conda activate`한 셸의 **프로젝트 루트**에서 실행합니다. 활성화하지 않고 환경의 `python.exe`를 경로로 직접 호출하면 DLL 경로가 잡히지 않아 일부 연산(matplotlib, scikit-learn)이 조용히 종료될 수 있습니다.

### 환경 변수 (`.env`)

프로젝트 루트에 `.env` 파일을 만듭니다. 이 파일은 커밋하지 않습니다.

```dotenv
# LLM 공급자: groq(기본) 또는 nvidia
LLM_PROVIDER=groq

# Groq 사용 시
GROQ_API_KEY=...
GROQ_MODEL=...          # 필수. 코드에 기본값이 없습니다

# NVIDIA 사용 시
NVIDIA_API_KEY=...
NVIDIA_MODEL=...        # 필수
NVIDIA_BASE_URL=...     # 선택. 기본 https://integrate.api.nvidia.com/v1
NVIDIA_STRUCTURED=prompt  # 선택. prompt | json_schema | guided_json

# 임베딩
EMBED_DEVICE=cpu        # 선택. 기본 cpu
HF_HUB_OFFLINE=1        # 기본 1(캐시된 모델만 사용). 최초 모델 다운로드 시 0

# 수집 (선택)
WIKIDOCS_DELAY_SEC=10
WIKIDOCS_USER_AGENT=...
```

> `bge-m3` 모델은 최초 1회 내려받아야 합니다. 기본값(`HF_HUB_OFFLINE=1`)에서는 외부 요청을 보내지 않으므로, 모델이 없다면 `HF_HUB_OFFLINE=0`으로 설정해 한 번 실행해 주세요.

### 처음 받았을 때 (데이터 준비)

`.data/`(수집 HTML·정제본·벡터 저장소)는 커밋하지 않으므로, 저장소를 처음 받으면 책 데이터가 없습니다. 이 상태에서 `python embed_pipeline.py`만 실행하면 ".data에 bookN 폴더가 없습니다" 오류로 끝납니다. 아래 순서로 준비합니다.

```bash
# 1) .env 작성 (위 예시). 최초 1회는 임베딩 모델을 받도록 HF_HUB_OFFLINE=0

# 2) 두 책 수집 → 정제 → 임베딩 (위키독스 외부 요청 발생, 요청 간격 기본 10초라 시간이 걸립니다)
python embed_pipeline.py --crawl --book-id 1      # 점프 투 파이썬
python embed_pipeline.py --crawl --book-id 110    # 파이썬으로 배우는 알고리즘 트레이딩

# 3) 모델을 받은 뒤에는 .env를 HF_HUB_OFFLINE=1로 되돌려 외부 요청을 막습니다

# 4) 확인 (답변 생성에 LLM을 1번 호출합니다)
python quiz_session.py "문자열 공백 제거는 어떻게 해?" --no-quiz
```

이후에는 캐시를 쓰므로 `python embed_pipeline.py`만으로 갱신됩니다.

## 사용법

### 1. 데이터 준비 (원클릭 파이프라인)

```bash
python embed_pipeline.py                          # .data의 모든 책: 수집(캐시) → 정제 → 임베딩
python embed_pipeline.py --book-id 110            # 특정 책만
python embed_pipeline.py --from embed             # 문단 임베딩부터
python embed_pipeline.py --to refine              # 정제까지만
python embed_pipeline.py --rebuild --from refine  # 건너뛰지 않고 정제부터 재실행
python embed_pipeline.py --rebuild --from embed   # 벡터 저장소를 archive로 옮기고 재생성
python embed_pipeline.py --crawl --book-id 2      # 새 책 수집 (위키독스 외부 요청 발생)
```

- 수집 단계는 기본적으로 **캐시된 HTML만** 사용합니다. 외부 요청이 필요한 신규 수집은 `--crawl`을 명시해야 합니다.
- `--rebuild`는 기존 산출물을 삭제하지 않고 `archive/`로 옮기며, 임베딩 텍스트가 동일한 문단의 벡터는 재사용합니다.
- 각 단계가 실패하면 뒷 단계가 오래된 데이터로 실행되지 않도록 파이프라인을 즉시 중단합니다.

단계별로 개별 실행도 가능합니다.

```bash
python text_ingestion.py --book-id 1              # 수집 (캐시 우선, 요청 간격 기본 10초)
python text_ingestion.py --offline                # 네트워크 없이 캐시만 사용
python text_ingestion.py --url https://wikidocs.net/13
python text_refine.py --book-id 1 [--force]       # 정제
python text_paragraphs.py --book-id 1 [--rebuild] # 문단 임베딩
python text_paragraphs.py --query "변수" --book-id 110   # 문단 검색 확인
```

정제 규칙은 프로젝트 루트의 `refine_config.json`(선택)으로 책별 조정할 수 있습니다.

```json
{
  "book1": {
    "exclude_titles": ["되새김 문제"],
    "exclude_ids": [180361],
    "include_ids": [4307]
  }
}
```

### 2. 질문 → 답변 → 확인 퀴즈

```bash
python quiz_session.py "문자열 공백 제거는 어떻게 해?"
python quiz_session.py "리스트 슬라이싱" --book-id 1 --k 4
python quiz_session.py "딕셔너리" --quiz 6 --solve              # 터미널에서 직접 풀기
python quiz_session.py "문자열 공백 제거" --formats ox,blank --quiz 4 --seed 7
python quiz_session.py "튜플" --no-quiz                          # 답변만
```

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `question` | (필수) | 학습자 질문 |
| `--k` | 5 | 근거 후보 문단 수(최대) |
| `--book-id` | 전체 | 검색할 책 제한 |
| `--max-distance` | 0.45 | 근거 거리 상한 |
| `--margin` | 0.15 | 1위 대비 허용 거리 차 |
| `--formats` | 전체 | 허용 유형(`ox,blank,mcq,short`). 하나만 주면 그 유형으로 고정 |
| `--quiz` | 4 | 총 문제 수 |
| `--seed` | 없음 | 문단 선택·보기 섞기 시드(재현용) |
| `--no-quiz` | - | 퀴즈 생략 |
| `--solve` | - | 직접 풀기. 정답 확인 전 확신도 입력 |

출력은 `=== 답변 ===`, `=== 출처 ===`(문단 경로·URL·거리), `=== 확인 퀴즈 ===` 순서입니다. 각 문제 아래에는 정답, 해설, 근거 문단과 URL이 함께 표시됩니다. `--solve`에서는 O/X와 4지선다는 자동 채점하고, 빈칸·단답형은 허용 답안 목록과 대조하되 판단이 어려우면 학습자가 직접 비교하도록 안내합니다.

> `evals/`, `demo/`, `tools/` 아래 파일은 루트의 모듈을 import하므로 **프로젝트 루트에서 `python -m`으로** 실행합니다.

### 3. 시연 GUI (Streamlit)

학습(질문 → 답변·근거) → 확인 퀴즈(직접 풀기) → 정답과 해설(+근거 URL) 순서로 진행하는 화면입니다. 질문 버튼 하나로 답변과 확인 문제(빈칸 1문제, 시드 0.5 고정)까지 만들고, 답을 제출하면 바로 채점과 해설을 보여 줍니다. 정답 목록에 없는 답은 LLM이 정답과 같은 뜻인지 판정합니다. `streamlit` 패키지가 필요합니다.

Streamlit 설정은 `.streamlit/config.toml`에 있습니다. 파일 감시기(`fileWatcherType = "none"`)를 끄는데, 감시기가 transformers 모듈을 훑다가 `No module named 'torchvision'` 로그를 내기 때문입니다. 코드를 고친 뒤에는 앱을 다시 실행하세요.

```powershell
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"; python -m streamlit run demo/demo_app.py
```

환경변수는 Streamlit을 띄우기 **전에** 설정해야 합니다. Streamlit이 먼저 protobuf를 불러오기 때문에, chromadb(opentelemetry-proto) 호환을 위한 코드 안의 우회 설정이 늦게 적용되어 `TypeError: Descriptors cannot be created directly`가 납니다.

### 4. 임베딩 시각화

```bash
python -m tools.text_visualize show --page-id 13 --csv .data/inspect/page13.csv
python -m tools.text_visualize heatmap --query "문자열 공백 제거" --open
python -m tools.text_visualize similarity --page-id 13 --open
python -m tools.text_visualize map --book-id 1 --query "슬라이싱" --open     # 2D·3D 산점도
```

결과는 `.data/inspect/`에 저장됩니다. 대상 지정은 `--query`, `--page-id`, `--ids` 중 하나를 사용합니다.

### 5. 정확도 평가 (개발용)

RAG on/off 조건에서 빈칸 문제의 품질을 비교하는 도구입니다. 케이스는 `.idea_folder/testcase/테스트 케이스.정확도 - 시트1.csv`(열: `id`, `book_id`, `개념`)를 사용합니다.

> **테스트 케이스 파일은 저장소에 포함되어 있지 않습니다.** `.idea_folder/`는 기획·메모 폴더라 커밋하지 않으므로, 케이스 CSV는 별도로 받아 위 경로에 두어야 합니다. 파일이 없으면 평가를 실행할 수 없습니다.

```bash
python -m evals.eval_blank --dry-run                     # LLM 없이 검색·book_id 필터 검사
python -m evals.eval_blank --ids 1 16 18                 # 일부 케이스만
python -m evals.eval_blank --skip-off --ids 1 3 4        # RAG off(대조군) 생략
python -m evals.eval_blank --context on --ids 16         # 같은 페이지 맥락 제공
python -m evals.eval_blank --resume .data/eval/<결과>.csv  # 중단된 실행 이어하기
```

- **on**: 검색된 상위 문단에서 문제를 생성하고, 문제 문장만으로 풀이 LLM이 정답을 하나로 도출할 수 있는지 검증합니다.
- **off**: 문단 본문 없이 책 제목·근거 URL·개념만 주고 기억에 의존해 출제하게 한 대조군입니다.
- 결과는 `.data/eval/blank_{일시}_{모델}_ctx{on|off}.csv`에 한 줄씩 기록되며, 판정 열(근거 확인, 답 유효성, 거부 타당)은 사람이 채웁니다.
- LLM 하루 한도에 도달하면 그때까지의 결과를 저장하고 중단합니다. `--resume`으로 이어서 실행합니다.

## 프로젝트 구조

```
.
├── embed_pipeline.py     # 수집 → 정제 → 임베딩 원클릭 실행
├── text_ingestion.py     # 위키독스 목차·본문 수집, 캐시, archive 이동
├── text_refine.py        # raw HTML → 블록 단위 chunks.jsonl, 코드 블록 분리
├── text_paragraphs.py    # A+ 문단 생성, Chroma 저장·검색, 적합성·페이지 관문
├── text_embed.py         # bge-m3 로딩·임베딩, 블록 텍스트 변환
├── quiz_session.py       # 질문 키워드 검색 → 답변 → 출제 → 풀이 (CLI 진입점)
├── quiz_templates.py     # 유형별 스키마·검증·빈칸 생성·채점
├── llm_groq.py           # Groq 호출
├── llm_nvidia.py         # NVIDIA(OpenAI 호환) 호출
├── evals/                # 테스트: 정확도 평가 (python -m evals.eval_blank)
│   ├── eval_blank.py     #   RAG on/off 빈칸 정확도 평가
│   └── eval_common.py    #   평가 공용 (케이스 CSV, 모델 옵션, 예외)
├── demo/                 # 시연: Streamlit GUI (python -m streamlit run demo/demo_app.py)
│   └── demo_app.py
├── tools/                # 점검 도구 (python -m tools.text_visualize)
│   └── text_visualize.py #   벡터 확인 (show / heatmap / similarity / map)
├── tests/                # 단위 테스트 (python -m unittest, LLM 불필요)
│   └── test_quiz_templates.py #   빈칸 문제 만들기
├── environment.yml       # conda 환경 정의 (검증한 버전)
├── .streamlit/
│   └── config.toml       # 데모용 Streamlit 설정 (파일 감시기 끄기)
├── .env                  # API 키·모델 설정 (커밋 제외, 직접 작성)
├── refine_config.json    # (선택) 책별 정제 예외 규칙. 없으면 기본 규칙
├── .idea_folder/         # 기획·메모 (커밋 제외). testcase/에 평가 케이스 CSV
├── .data/                # 수집·정제·벡터 산출물 (커밋 제외)
└── archive/              # 교체·폐기된 파일 보관: {종류}/{날짜}/{파일명_날짜} (커밋 제외)
```

### 데이터 레이아웃 (`.data/`)

```
.data/
├── book{N}/
│   ├── raw/{page_id}.html    # 수집 원본 캐시
│   ├── pages.jsonl           # 목차·페이지 메타
│   ├── chunks.jsonl          # 정제된 블록 (페이지 1개 = 1행)
│   ├── code_blocks.jsonl     # 코드 블록 저장소
│   ├── excluded.json         # 자동 제외된 페이지와 사유
│   └── failed.json           # 수집에 실패한 페이지 (실패가 있을 때만)
├── vectorstore_paragraphs/   # Chroma (컬렉션 paragraphs_bge-m3, cosine)
├── eval/                     # 정확도 평가 결과 CSV (평가 실행 시 생성)
└── inspect/                  # 시각화 산출물 (시각화 실행 시 생성)
```

## 설계 메모

- **문단 구성(A+)**: 설명 블록(`paragraph`, `note`, `key_point`, `caption`, `concept_box`, 연속 `list`)과 직후의 코드·표를 한 묶음으로 보고, 같은 소제목 안에서 200자 목표(상한 600자)까지 이웃과 병합합니다. 문단 ID는 `{chunk_id}#b{시작 블록}` 형식이라 앞 문단이 바뀌어도 뒤 문단 ID가 밀리지 않습니다.
- **코드 처리**: LLM에 전달하는 본문과 페이지 맥락에서는 코드를 제외하고(`code_ids`로 코드 저장소와 연결), 임베딩 텍스트에는 앞부분 8줄을 포함합니다.
- **적합성 관문**: 도입·예고, 회고, 비유, 평가·감상 문장을 제외한 사실 문장이 40자 미만이면 출제 근거에서 제외합니다. 이 관문은 정확도 평가(`evals`)의 검색에만 적용되며, 앱 퀴즈(`quiz_session`)는 대신 LLM의 출제 거부(`reject_reason`)로 부적합한 문단을 거릅니다.
- **출제 거부**: 정보 없음 / 비유 / 다른 언어 / 추상적 주장 / 개념 불일치에 해당하면 LLM이 `reject_reason`으로 거부할 수 있으며, 거부된 문단은 재시도하지 않습니다.
- **검색 임계값**: 관련 질문 15개와 무관 질문 6개로 측정해 `MAX_DISTANCE=0.45`를 정했습니다. 이 값은 질문 문장으로 검색하던 때 측정한 값이라, 키워드 검색으로 바꾼 뒤에는 다시 측정이 필요합니다.
- **질문 키워드 검색**: 검색은 질문 문장이 아니라 키워드로 합니다(`quiz_session.question_keywords`). 문장을 그대로 넘기면 "어떻게", "무엇인가" 같은 단어가 페이지 관문(`page_gate`)에서 "06장 … 어떻게 시작해야 할까?", "01장 파이썬이란 무엇인가?" 같은 장 제목에 걸려 무관한 페이지만 남고, 근거가 0개가 되었습니다(예: "변수란 무엇인가"는 전체 책에서 0개, 알고리즘 트레이딩만 고르면 5개).

## 안전 및 운영 원칙

- **외부 네트워크 요청 최소화**: 수집은 캐시 우선이며 신규 요청은 `--crawl`(또는 `text_ingestion.py`의 직접 실행)에서만 발생합니다. 위키독스가 403/429로 응답하면 우회하지 않고 즉시 중단합니다. 임베딩 모델도 기본적으로 캐시만 사용하고, Chroma 텔레메트리는 비활성화되어 있습니다.
- **삭제 대신 보관**: 재생성이 필요한 산출물은 삭제하지 않고 `archive/`로 이동하며 파일명에 일시를 붙입니다.
- **비밀 정보**: API 키는 `.env`로만 관리하며 커밋하지 않습니다.

## 개발

```bash
ruff check
ruff format
python -m unittest        # 단위 테스트 (LLM·벡터 저장소 없이 돈다)
```

예외 처리는 구체적인 예외 타입을 지정하며(bare `except:` 금지), 환경 변수는 `python-dotenv`로 읽습니다.

## 알려진 제약

- 현재 수집 대상은 위키독스 책입니다(`wikidocs.net`).
- 코드 출제는 보류 상태입니다. 정확도 평가에서는 코드가 포함된 문단을 적합성 관문에서 제외하고(`EXCLUDE_CODE = True`), 앱 퀴즈는 코드가 있던 문단도 쓰되 LLM에는 코드를 뺀 본문만 전달합니다.
- NVIDIA 공급자의 엔드포인트·구조화 출력·일일 한도 판정은 공식 문서로 검증하지 않은 일반적인 OpenAI 호환 방식 기준입니다.
- 단위 테스트(`tests/`)는 빈칸 문제 만들기만 다룹니다. 출제 품질 검증은 `evals/eval_blank.py`로 수행합니다.
