# check_your_learning_by_AI

학습자의 메타인지를 점검하고, 학습 직후 확인 퀴즈를 즉시 제공해 장기 기억 유지 효과를 높이는 AI 도구입니다.

학습자가 질문하면 교재 본문에서 근거 문단을 검색해 답변하고, **답변에 사용된 바로 그 문단**에서 퀴즈를 출제합니다. 모든 문제는 근거 문단과 출처 URL을 함께 제시하며, 문제와 정답이 실제로 문단에 존재하는지 코드가 한 번 더 검증합니다.

## 주요 특징

- **근거 기반 답변(RAG)**: 교재 본문 문단만을 근거로 답변하며, 문장마다 인용 번호(`[1]`, `[2]`)를 표기합니다. 자료에 답이 없으면 추측하지 않습니다.
- **문단 단위 인덱스**: 설명 블록과 코드·표를 묶은 200~600자 내외의 문단(A+ 문단)을 검색 단위로 사용합니다.
- **4가지 퀴즈 유형**: O/X, 빈칸, 4지선다, 단답형. 유형은 LLM이 문단 성격에 맞춰 문단별로 선택합니다.
- **출제 검증**: LLM이 제시한 근거 인용(`evidence`)과 정답이 문단 원문에 있는지, 보기 중복·정답 번호 범위·정답 노출 여부 등을 코드로 재검사하고, 실패하면 다른 문단으로 재시도합니다.
- **메타인지 점검**: 풀이 모드(`--solve`)에서 정답 확인 전 확신도(모름/애매/확실)를 먼저 입력합니다.
- **LLM 공급자 교체**: `.env`의 `LLM_PROVIDER` 값만 바꿔 Groq와 NVIDIA를 전환합니다.
- **증분 처리**: 내용 해시를 기준으로 변경된 파일·벡터만 갱신합니다.

## 동작 흐름

```
[위키독스] ──수집──▶ raw HTML ──정제──▶ chunks.jsonl ──문단화·임베딩──▶ Chroma(bge-m3)
 text_ingestion.py      text_refine.py       text_paragraphs.py
                                                      │
 학습자 질문 ──검색(거리 필터)──▶ 근거 문단 ──▶ 답변 + 인용
                                                      │
                          인용 문단 + 같은 페이지의 다른 문단 ──▶ 문제 출제 ──▶ 검증 ──▶ 풀이·채점
                                        quiz_session.py / quiz_templates.py
```

1. **수집**: 위키독스 책의 목차와 페이지 HTML을 `.data/book{N}/raw/`에 캐시합니다.
2. **정제**: HTML을 블록(문단, 코드, 표, 목록 등) 단위로 구조화해 `chunks.jsonl`(페이지 1개 = 1행)로 저장합니다. 안내·목차·부속 페이지는 자동 제외하며, 코드 블록은 `code_blocks.jsonl`에 별도 저장합니다.
3. **문단화·임베딩**: `chunks.jsonl`에서 A+ 문단을 생성하고, `BAAI/bge-m3`로 임베딩해 Chroma에 저장합니다.
4. **검색·답변**: 질문과 코사인 거리가 가까운 문단만(`MAX_DISTANCE=0.45`, 1위 대비 `0.15` 이내) 근거로 채택합니다. 무관한 질문은 LLM 호출 없이 종료됩니다.
5. **출제**: 답변이 인용한 문단과 같은 페이지의 다른 문단을 약 2:1로 섞어 문제를 만듭니다.

## 요구 사항

| 항목 | 내용 |
| --- | --- |
| 언어/런타임 | Python 3.13 |
| 패키지 관리 | conda |
| 린트·포맷 | Ruff |
| 실행 환경 | 로컬 |

주요 의존 패키지: `python-dotenv`, `beautifulsoup4`, `numpy`, `sentence-transformers`, `chromadb`, `httpx`, `groq`

시각화(`text_visualize.py`) 선택 패키지: `matplotlib`, `pandas`, `scikit-learn`, `plotly`, `umap-learn`

> 이 저장소에는 `requirements.txt`나 `environment.yml`이 없습니다. 위 패키지를 conda 환경에 직접 설치해 주세요.

## 설치 및 설정

```bash
# 1) conda 환경 생성 (예시)
conda create -n check-learning python=3.13
conda activate check-learning

# 2) 의존 패키지 설치 (conda-forge 기준 예시)
conda install -c conda-forge python-dotenv beautifulsoup4 numpy sentence-transformers chromadb httpx groq
```

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

### 3. 임베딩 시각화

```bash
python text_visualize.py show --page-id 13 --csv .data/inspect/page13.csv
python text_visualize.py heatmap --query "문자열 공백 제거" --open
python text_visualize.py similarity --page-id 13 --open
python text_visualize.py map --book-id 1 --query "슬라이싱" --open     # 2D·3D 산점도
```

결과는 `.data/inspect/`에 저장됩니다. 대상 지정은 `--query`, `--page-id`, `--ids` 중 하나를 사용합니다.

### 4. 정확도 평가 (개발용)

RAG on/off 조건에서 빈칸 문제의 품질을 비교하는 도구입니다. 케이스는 `.idea_folder/testcase/`의 CSV를 사용합니다.

```bash
python eval_blank.py --dry-run                     # LLM 없이 검색·book_id 필터 검사
python eval_blank.py --ids 1 16 18                 # 일부 케이스만
python eval_blank.py --skip-off --ids 1 3 4        # RAG off(대조군) 생략
python eval_blank.py --context on --ids 16         # 같은 페이지 맥락 제공
python eval_blank.py --resume .data/eval/<결과>.csv  # 중단된 실행 이어하기
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
├── text_visualize.py     # 벡터 확인 (show / heatmap / similarity / map)
├── quiz_session.py       # 검색 → 답변 → 출제 → 풀이 (CLI 진입점)
├── quiz_templates.py     # 유형별 스키마·검증·빈칸 생성·채점
├── llm_groq.py           # Groq 호출
├── llm_nvidia.py         # NVIDIA(OpenAI 호환) 호출
├── eval_blank.py         # RAG on/off 빈칸 정확도 평가
├── eval_common.py        # 평가 공용 (케이스 CSV, 모델 옵션, 예외)
├── .data/                # 수집·정제·벡터 산출물 (커밋 제외)
└── archive/              # 교체·폐기된 파일 보관 (날짜 접미사)
```

### 데이터 레이아웃 (`.data/`)

```
.data/
├── book{N}/
│   ├── raw/{page_id}.html    # 수집 원본 캐시
│   ├── pages.jsonl           # 목차·페이지 메타
│   ├── chunks.jsonl          # 정제된 블록 (페이지 1개 = 1행)
│   ├── code_blocks.jsonl     # 코드 블록 저장소
│   └── excluded.json         # 자동 제외된 페이지와 사유
├── vectorstore_paragraphs/   # Chroma (컬렉션 paragraphs_bge-m3, cosine)
├── eval/                     # 정확도 평가 결과 CSV
└── inspect/                  # 시각화 산출물
```

## 설계 메모

- **문단 구성(A+)**: 설명 블록(`paragraph`, `note`, `key_point`, `caption`, `concept_box`, 연속 `list`)과 직후의 코드·표를 한 묶음으로 보고, 같은 소제목 안에서 200자 목표(상한 600자)까지 이웃과 병합합니다. 문단 ID는 `{chunk_id}#b{시작 블록}` 형식이라 앞 문단이 바뀌어도 뒤 문단 ID가 밀리지 않습니다.
- **코드 처리**: LLM에 전달하는 본문과 페이지 맥락에서는 코드를 제외하고(`code_ids`로 코드 저장소와 연결), 임베딩 텍스트에는 앞부분 8줄을 포함합니다.
- **적합성 관문**: 도입·예고, 회고, 비유, 평가·감상 문장을 제외한 사실 문장이 40자 미만이면 출제 근거에서 제외합니다.
- **출제 거부**: 정보 없음 / 비유 / 다른 언어 / 추상적 주장 / 개념 불일치에 해당하면 LLM이 `reject_reason`으로 거부할 수 있으며, 거부된 문단은 재시도하지 않습니다.
- **검색 임계값**: 관련 질문 15개와 무관 질문 6개로 측정해 `MAX_DISTANCE=0.45`를 정했습니다. 표현 범위가 넓은 질문(예: "변수란 무엇인가")은 근거를 찾지 못할 수 있습니다.

## 안전 및 운영 원칙

- **외부 네트워크 요청 최소화**: 수집은 캐시 우선이며 신규 요청은 `--crawl`(또는 `text_ingestion.py`의 직접 실행)에서만 발생합니다. 위키독스가 403/429로 응답하면 우회하지 않고 즉시 중단합니다. 임베딩 모델도 기본적으로 캐시만 사용하고, Chroma 텔레메트리는 비활성화되어 있습니다.
- **삭제 대신 보관**: 재생성이 필요한 산출물은 삭제하지 않고 `archive/`로 이동하며 파일명에 일시를 붙입니다.
- **비밀 정보**: API 키는 `.env`로만 관리하며 커밋하지 않습니다.

## 개발

```bash
ruff check
ruff format
```

예외 처리는 구체적인 예외 타입을 지정하며(bare `except:` 금지), 환경 변수는 `python-dotenv`로 읽습니다.

## 알려진 제약

- 현재 수집 대상은 위키독스 책입니다(`wikidocs.net`).
- 코드가 포함된 문단은 현재 적합성 관문에서 기본 제외됩니다(`EXCLUDE_CODE = True`). 코드 출제는 보류 상태입니다.
- NVIDIA 공급자의 엔드포인트·구조화 출력·일일 한도 판정은 공식 문서로 검증하지 않은 일반적인 OpenAI 호환 방식 기준입니다.
- 별도의 테스트 스위트는 아직 없습니다. 품질 검증은 `eval_blank.py`로 수행합니다.
