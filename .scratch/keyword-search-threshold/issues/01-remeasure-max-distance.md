# 키워드 검색 기준으로 MAX_DISTANCE 다시 재기

Status: needs-triage

## 문제

`quiz_session.MAX_DISTANCE=0.45`와 `DISTANCE_MARGIN=0.15`는 질문 문장으로 검색하던 때 관련 질문 15개·무관 질문 6개로 잰 값이에요. 지금은 키워드로 검색하므로(`docs/adr/0004-keyword-search.md`) 거리 분포가 달라졌을 수 있어요 (README "설계 메모").

## 할 일 (안)

- 같은 21개 질문을 키워드 검색으로 다시 돌려 관련 질문의 근거 수와 무관 질문 차단을 확인해요.
- 측정은 검색만 하므로 LLM 호출이 필요 없어요.

## Comments
