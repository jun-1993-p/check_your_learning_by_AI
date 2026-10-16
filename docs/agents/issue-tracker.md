# Issue tracker: Local Markdown

이 저장소의 이슈와 명세는 `.scratch/` 아래 Markdown 파일이에요. `.scratch/`는 git으로 추적해요.

## Conventions

- 기능 하나 = 폴더 하나: `.scratch/<feature-slug>/`. slug는 영어 kebab-case이고, 작업 브랜치 이름과 같아요.
- 명세: `.scratch/<feature-slug>/spec.md`
- 구현 티켓: 티켓 하나당 파일 하나 `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, `01`부터 번호를 매겨요. 여러 티켓을 한 파일에 합치지 않아요.
- 상태는 파일 위쪽의 `Status:` 줄에 적어요 (`**Status:**` 굵은 형식도 같은 뜻). 값은 `triage-labels.md`의 라벨이거나, 끝난 작업이면 `closed`예요.
- 댓글과 대화 기록은 파일 맨 아래 `## Comments` 제목 밑에 이어 붙여요.
- 본문은 한국어로 써요. 위의 줄 이름(`Status:`, `## Comments`, `## Answer` 등)은 그대로 둬요.

## When a skill says "publish to the issue tracker"

`.scratch/<feature-slug>/` 아래에 새 파일을 만들어요 (폴더가 없으면 만들어요). 새 이슈는 `Status: needs-triage`로 시작해요.

## When a skill says "fetch the relevant ticket"

참조된 경로의 파일을 읽어요. 사용자는 보통 경로나 이슈 번호를 직접 알려 줘요.

## Closing work

티켓을 끝내면 `Status: closed`로 바꾸고, `## Comments`에 브랜치 이름과 커밋 해시를 적어요. 폴더나 파일은 지우지 않아요 (archive 규칙).

## Wayfinding operations

Used by `/wayfinder`. The **map** is a file with one **child** file per ticket.

- **Map**: `.scratch/<effort>/map.md` (the Notes / Decisions-so-far / Fog body).
- **Child ticket**: `.scratch/<effort>/issues/NN-<slug>.md`, numbered from `01`, with the question in the body. A `Type:` line records the ticket type (`research`/`prototype`/`grilling`/`task`); a `Status:` line records `claimed`/`resolved`.
- **Blocking**: a `Blocked by: NN, NN` line near the top. A ticket is unblocked when every file it lists is `resolved`.
- **Frontier**: scan `.scratch/<effort>/issues/` for files that are open, unblocked, and unclaimed; first by number wins.
- **Claim**: set `Status: claimed` and save before any work.
- **Resolve**: append the answer under an `## Answer` heading, set `Status: resolved`, then append a context pointer (gist + link) to the map's Decisions-so-far in `map.md`.
