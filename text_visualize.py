"""Chroma에 저장한 문단 임베딩 벡터를 눈으로 확인한다.

하위 명령:
    show        벡터 숫자 값·길이·통계를 출력하고, 원하면 CSV로 저장
    heatmap     벡터를 행으로 쌓아 1024차원 값을 색으로 그린다 (PNG)
    similarity  벡터끼리의 코사인 유사도 행렬을 그린다 (PNG)
    map         전체 벡터를 2D·3D로 줄인 산점도 (plotly가 있으면 HTML, 없으면 2D PNG)

show/heatmap/similarity의 대상 고르기 (하나만 지정, 없으면 --book-id·--limit 범위):
    --query "질문"   검색 상위 문단 + 질의 벡터 (질의는 맨 위 행)
    --page-id 13     한 페이지의 문단들
    --ids 13-01#b000 13-01#b010   (문단 id = {페이지}-01#b{시작 블록 번호})

사용 예:
    python text_visualize.py show --ids 13-01#b000 13-01#b010 --n 10
    python text_visualize.py show --page-id 13 --csv .data/inspect/page13.csv
    python text_visualize.py heatmap --query "문자열 공백 제거" --open
    python text_visualize.py heatmap --page-id 13 --sort-dims --dims 64
    python text_visualize.py similarity --page-id 13 --open
    python text_visualize.py map --book-id 1 --query "슬라이싱" --open
    python text_visualize.py map --dim 3 --color book --query "이동평균" --open
"""

import argparse
import logging
import webbrowser
from pathlib import Path

import numpy as np

from text_embed import DATA_ROOT, embed, load_model
from text_paragraphs import ParagraphIndex, get_collection

OUT_DIR = DATA_ROOT / "inspect"
SNIPPET_CHARS = 160
LABEL_CHARS = 30
ANNOTATE_MAX = 25  # 이 이하일 때만 유사도 행렬 칸에 숫자를 쓴다
RANDOM_STATE = 42
KOREAN_FONTS = ["Malgun Gothic", "NanumGothic", "AppleGothic"]

logger = logging.getLogger("text_visualize")


# ---------------------------------------------------------------- 벡터 고르기


def build_where(**conditions) -> dict | None:
    filters = [{k: v} for k, v in conditions.items() if v is not None]
    if not filters:
        return None
    return filters[0] if len(filters) == 1 else {"$and": filters}


def fetch(collection, book_id: int | None) -> dict:
    where = build_where(book_id=book_id)
    return collection.get(where=where, include=["embeddings", "metadatas", "documents"])


def sort_key(item: tuple) -> tuple:
    """페이지 → 조각 순번 → 문단 순서로 문서 흐름대로 놓는다."""
    vid, _, meta = item
    _, _, seq = meta["chunk_id"].partition("-")
    seq_no = int(seq) if seq.isdigit() else 0
    return (meta.get("page_id", 0), seq_no, vid)


def select_vectors(args, collection) -> tuple[list[str], np.ndarray, list[dict]]:
    """show/heatmap/similarity 대상. 질의가 있으면 질의 벡터를 맨 앞에 붙인다."""
    include = ["embeddings", "metadatas"]
    order: list[str] | None = None

    if args.ids:
        got = collection.get(ids=args.ids, include=include)
        order = args.ids
    elif args.query:
        model = load_model()
        hits = ParagraphIndex(model, collection).find(args.query, args.book_id)
        order = [h["unit_id"] for h in hits[: args.limit]]  # 검색 순위대로 놓는다
        logger.info("검색 결과: %s", order)
        got = collection.get(ids=order, include=include)
    else:
        where = build_where(book_id=args.book_id, page_id=args.page_id)
        got = collection.get(where=where, limit=args.limit, include=include)

    items = list(zip(got["ids"], got["embeddings"], got["metadatas"]))
    if args.ids or args.query:
        items.sort(key=lambda it: order.index(it[0]))
    else:
        items.sort(key=sort_key)
    if not items:
        raise SystemExit("조건에 맞는 벡터가 없습니다")

    ids = [it[0] for it in items]
    vectors = np.asarray([it[1] for it in items], dtype=np.float32)
    metas = [it[2] for it in items]
    if args.query:
        ids.insert(0, "질의")
        metas.insert(0, {"unit_id": "질의", "kind": "query", "section": args.query})
        vectors = np.vstack([np.asarray(embed(model, [args.query])), vectors])
    return ids, vectors, metas


def row_label(meta: dict) -> str:
    if meta.get("kind") == "query":
        text = f"★ 질의: {meta['section']}"
    else:
        text = f"■ {meta['unit_id']} {meta.get('section', '')}"
    return text if len(text) <= LABEL_CHARS else text[: LABEL_CHARS - 1] + "…"


# ---------------------------------------------------------------- matplotlib


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장한다
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in KOREAN_FONTS if f in installed), None)
    if font:
        plt.rcParams["font.family"] = font
    else:
        logger.warning("한글 글꼴을 찾지 못해 라벨이 깨질 수 있습니다")
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def save_figure(plt, fig, out: Path, open_after: bool) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("저장: %s", out)
    if open_after:
        webbrowser.open(out.resolve().as_uri())


# ---------------------------------------------------------------- 하위 명령


def cmd_show(args, collection) -> None:
    ids, vectors, metas = select_vectors(args, collection)
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    for vid, vec, meta in zip(ids, vectors, metas):
        print(f"\n{vid}  {row_label(meta)}")
        print(f"  앞 {args.n}개: {vec[: args.n]}")
        print(
            f"  길이 {np.linalg.norm(vec):.4f}  최소 {vec.min():.4f}  "
            f"최대 {vec.max():.4f}  평균 {vec.mean():.5f}  "
            f"|값|>0.1인 차원 {(np.abs(vec) > 0.1).sum()}개"
        )
    if len(vectors) > 1:
        sims = vectors[0] @ vectors[1:].T
        print(f"\n{ids[0]} 기준 코사인 유사도:")
        for vid, sim in zip(ids[1:], sims):
            print(f"  {sim:.4f}  (거리 {1 - sim:.4f})  {vid}")
    if args.csv:
        import pandas as pd

        args.csv.parent.mkdir(parents=True, exist_ok=True)
        columns = [f"d{i}" for i in range(vectors.shape[1])]
        frame = pd.DataFrame(vectors, index=ids, columns=columns)
        frame.insert(0, "label", [row_label(m) for m in metas])
        frame.to_csv(args.csv, encoding="utf-8-sig")  # 엑셀에서 한글이 깨지지 않게
        logger.info("CSV 저장: %s (%d행 × %d차원)", args.csv, *vectors.shape)


def cmd_heatmap(args, collection) -> None:
    ids, vectors, metas = select_vectors(args, collection)
    data = vectors
    if args.sort_dims:
        # 고른 벡터들 사이에서 값이 크게 갈리는 차원을 왼쪽에 모은다
        data = data[:, np.argsort(-data.var(axis=0))]
    data = data[:, : args.dims]  # 정렬 후 자르면 --dims는 "상위 N개 차원"이 된다

    plt = setup_matplotlib()
    limit = float(np.percentile(np.abs(data), 99))  # 극단값 몇 개에 색이 묻히지 않게
    fig, ax = plt.subplots(figsize=(14, 1.5 + 0.3 * len(ids)))
    image = ax.imshow(
        data,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    ax.set_yticks(range(len(ids)), [row_label(m) for m in metas], fontsize=8)
    ax.set_xlabel(
        "차원 (분산 큰 순서)" if args.sort_dims else f"차원 0 ~ {data.shape[1] - 1}"
    )
    ax.set_title(f"임베딩 벡터 히트맵 ({len(ids)}개 × {data.shape[1]}차원)")
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01, label="값")
    save_figure(plt, fig, args.out or OUT_DIR / "heatmap.png", args.open)


def cmd_similarity(args, collection) -> None:
    ids, vectors, metas = select_vectors(args, collection)
    if len(ids) < 2:
        raise SystemExit("유사도를 비교하려면 벡터가 2개 이상 필요합니다")
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    sims = unit @ unit.T
    off_diag = sims[~np.eye(len(ids), dtype=bool)]

    plt = setup_matplotlib()
    size = 3 + 0.4 * len(ids)
    fig, ax = plt.subplots(figsize=(size + 2, size))
    image = ax.imshow(sims, cmap="viridis", vmin=float(off_diag.min()), vmax=1.0)
    labels = [row_label(m) for m in metas]
    ax.set_xticks(range(len(ids)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(ids)), labels, fontsize=8)
    if len(ids) <= ANNOTATE_MAX:
        threshold = (off_diag.min() + 1) / 2
        for i in range(len(ids)):
            for j in range(len(ids)):
                ax.text(
                    j,
                    i,
                    f"{sims[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if sims[i, j] > threshold else "white",
                )
    ax.set_title(
        f"코사인 유사도 행렬 ({len(ids)}개, 대각선 제외 "
        f"{off_diag.min():.2f} ~ {off_diag.max():.2f})"
    )
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, label="코사인 유사도")
    save_figure(plt, fig, args.out or OUT_DIR / "similarity.png", args.open)


# ---------------------------------------------------------------- 2D·3D 지도


def reduce(vectors: np.ndarray, method: str, dim: int = 2) -> np.ndarray:
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=dim, random_state=RANDOM_STATE).fit_transform(vectors)
    if method == "umap":
        try:
            import umap
        except ImportError as e:
            raise SystemExit(
                "umap-learn이 없습니다. --method tsne 또는 pca를 쓰세요"
            ) from e
        return umap.UMAP(
            n_components=dim, metric="cosine", random_state=RANDOM_STATE
        ).fit_transform(vectors)

    from sklearn.manifold import TSNE

    perplexity = min(30, max(5, (len(vectors) - 1) // 3))
    return TSNE(
        n_components=dim,
        metric="cosine",
        init="pca",
        perplexity=perplexity,
        random_state=RANDOM_STATE,
    ).fit_transform(vectors)


def color_label(meta: dict, color: str) -> str:
    if color == "book":
        return meta.get("source_dir", str(meta.get("book_id", "")))
    chapter = meta.get("path", "").split(" > ")[0]
    return f"{meta.get('source_dir', '')} | {chapter}"


def snippet(text: str) -> str:
    # 첫 줄은 [경로] 머리말이라 hover에서 중복되므로 뺀다
    body = " ".join(text.split("\n")[1:]).strip()
    body = body[:SNIPPET_CHARS] + ("…" if len(body) > SNIPPET_CHARS else "")
    return body.replace("<", "&lt;")


def scatter(go, pts: np.ndarray, **kwargs):
    """좌표 차원 수에 맞는 plotly 산점도 trace를 만든다."""
    if pts.shape[1] == 3:
        return go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], **kwargs)
    return go.Scattergl(x=pts[:, 0], y=pts[:, 1], **kwargs)


def build_figure(points, metas, docs, color, query_points, queries, hit_ids):
    import plotly.graph_objects as go

    is_3d = points.shape[1] == 3
    scale = 0.6 if is_3d else 1.0  # 3D 마커는 같은 크기여도 더 크게 보인다
    fig = go.Figure()
    labels = [color_label(m, color) for m in metas]
    for label in sorted(set(labels)):
        idx = [i for i, lab in enumerate(labels) if lab == label]
        fig.add_trace(
            scatter(
                go,
                points[idx],
                mode="markers",
                name=label,
                marker={"size": 8 * scale, "opacity": 0.8},
                customdata=[
                    [
                        metas[i]["unit_id"],
                        metas[i].get("path", ""),
                        metas[i].get("section", ""),
                        snippet(docs[i]),
                    ]
                    for i in idx
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{customdata[1]}<br><i>%{customdata[2]}</i><br>"
                    "%{customdata[3]}<extra></extra>"
                ),
            )
        )

    # 검색 상위 결과는 테두리로 강조한다
    hit_idx = [i for i, m in enumerate(metas) if m["unit_id"] in hit_ids]
    if hit_idx:
        fig.add_trace(
            scatter(
                go,
                points[hit_idx],
                mode="markers",
                name="검색 결과",
                marker={
                    "size": 16 * scale,
                    "symbol": "circle-open",
                    "color": "black",
                    "line": {"width": 2, "color": "black"},
                },
                hoverinfo="skip",
            )
        )
    if queries:
        fig.add_trace(
            scatter(
                go,
                query_points,
                mode="markers+text",
                name="질의",
                text=queries,
                textposition="top center",
                # 3D 산점도는 star 기호를 지원하지 않는다
                marker={
                    "size": 18 * scale,
                    "symbol": "diamond" if is_3d else "star",
                    "color": "red",
                },
                hovertemplate="질의: %{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"임베딩 지도 {'3D' if is_3d else '2D'} (벡터 {len(metas)}개)",
        legend={"itemsizing": "constant", "font": {"size": 10}},
        hoverlabel={"align": "left"},
        template="plotly_white",
        height=850,
    )
    if is_3d:
        hidden = {"showticklabels": False, "title": ""}
        fig.update_scenes(xaxis=hidden, yaxis=hidden, zaxis=hidden)
    else:
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
    return fig


def build_static_map(plt, points, metas, color, query_points, queries, hit_ids):
    """plotly가 없을 때 쓰는 정적 산점도. hover 대신 범례와 질의 라벨만 있다."""
    fig, ax = plt.subplots(figsize=(13, 10))
    labels = [color_label(m, color) for m in metas]
    groups = sorted(set(labels))
    cmap = plt.get_cmap("tab20", max(len(groups), 1))
    for n, label in enumerate(groups):
        idx = [i for i, lab in enumerate(labels) if lab == label]
        ax.scatter(
            points[idx, 0],
            points[idx, 1],
            s=18,
            marker="o",
            color=cmap(n),
            alpha=0.75,
            label=label,
        )
    hit_idx = [i for i, m in enumerate(metas) if m["unit_id"] in hit_ids]
    if hit_idx:
        ax.scatter(
            points[hit_idx, 0],
            points[hit_idx, 1],
            s=160,
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="검색 결과",
        )
    for (x, y), q in zip(query_points, queries):
        ax.scatter([x], [y], s=300, marker="*", color="red", zorder=5)
        ax.annotate(q, (x, y), textcoords="offset points", xytext=(6, 6), color="red")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"임베딩 지도 (벡터 {len(metas)}개)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7, frameon=False)
    return fig


def cmd_map(args, collection) -> None:
    got = fetch(collection, args.book_id)
    if len(got["ids"]) < 3:
        raise SystemExit("시각화할 벡터가 부족합니다. 필터 조건을 확인하세요")
    vectors = np.asarray(got["embeddings"], dtype=np.float32)
    logger.info("벡터 %d개, %s로 %d차원 축소", len(vectors), args.method, args.dim)

    hit_ids: set[str] = set()
    if args.query:
        model = load_model()
        # t-SNE는 새 점을 따로 변환할 수 없어서 질의 벡터를 함께 넣어 축소한다
        vectors = np.vstack([vectors, np.asarray(embed(model, args.query))])
        finder = ParagraphIndex(model, collection)
        for q in args.query:
            hits = finder.find(q, args.book_id)[: args.k]
            hit_ids.update(h["unit_id"] for h in hits)
            logger.info("%s → %s", q, [h["unit_id"] for h in hits])

    points = reduce(vectors, args.method, args.dim)
    n = len(got["ids"])
    try:
        import plotly  # noqa: F401
    except ImportError:
        if args.dim == 3:
            raise SystemExit("3D 지도는 plotly가 필요합니다") from None
        logger.info("plotly가 없어 정적 PNG로 그립니다")
        plt = setup_matplotlib()
        fig = build_static_map(
            plt,
            points[:n],
            got["metadatas"],
            args.color,
            points[n:],
            args.query,
            hit_ids,
        )
        save_figure(plt, fig, args.out or OUT_DIR / "map.png", args.open)
        return

    fig = build_figure(
        points[:n],
        got["metadatas"],
        got["documents"],
        args.color,
        points[n:],
        args.query,
        hit_ids,
    )
    out = args.out or OUT_DIR / ("map3d.html" if args.dim == 3 else "map.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs=True)  # CDN 대신 내장
    logger.info("저장: %s", out)
    if args.open:
        webbrowser.open(out.resolve().as_uri())


# ---------------------------------------------------------------- 실행


def add_selection_args(p: argparse.ArgumentParser, default_limit: int) -> None:
    target = p.add_mutually_exclusive_group()
    target.add_argument("--query", help="검색 상위 문단 + 질의 벡터")
    target.add_argument("--page-id", type=int, help="한 페이지의 문단들")
    target.add_argument("--ids", nargs="+", help="벡터 ID 직접 지정")
    p.add_argument("--book-id", type=int, default=None)
    p.add_argument(
        "--limit",
        type=int,
        default=default_limit,
        help="최대 벡터 수 (--query면 검색 문단 수)",
    )


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description="임베딩 벡터 시각화")
    sub = parser.add_subparsers(dest="command", required=True)

    p_show = sub.add_parser("show", help="벡터 숫자 값 보기")
    add_selection_args(p_show, default_limit=5)
    p_show.add_argument("--n", type=int, default=8, help="출력할 앞쪽 값 개수")
    p_show.add_argument("--csv", type=Path, help="전체 값을 CSV로 저장")

    p_heat = sub.add_parser("heatmap", help="벡터 히트맵")
    add_selection_args(p_heat, default_limit=20)
    p_heat.add_argument(
        "--dims",
        type=int,
        default=1024,
        help="그릴 차원 수 (--sort-dims와 함께 쓰면 분산 상위 N개)",
    )
    p_heat.add_argument(
        "--sort-dims", action="store_true", help="값이 크게 갈리는 차원부터 정렬"
    )

    p_sim = sub.add_parser("similarity", help="코사인 유사도 행렬")
    add_selection_args(p_sim, default_limit=20)

    p_map = sub.add_parser("map", help="2D 산점도")
    p_map.add_argument("--book-id", type=int, default=None)
    p_map.add_argument("--method", choices=["tsne", "umap", "pca"], default="tsne")
    p_map.add_argument("--dim", type=int, choices=[2, 3], default=2, help="축소 차원")
    p_map.add_argument("--color", choices=["chapter", "book"], default="chapter")
    p_map.add_argument("--query", action="append", default=[], help="여러 번 지정 가능")
    p_map.add_argument("--k", type=int, default=5, help="질의별 강조할 검색 결과 수")

    for p in (p_show, p_heat, p_sim, p_map):
        if p is not p_show:
            p.add_argument("--out", type=Path, default=None, help="저장 경로")
            p.add_argument("--open", action="store_true", help="저장 후 바로 열기")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    commands = {
        "show": cmd_show,
        "heatmap": cmd_heatmap,
        "similarity": cmd_similarity,
        "map": cmd_map,
    }
    commands[args.command](args, get_collection())


if __name__ == "__main__":
    main()
