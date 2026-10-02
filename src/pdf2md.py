"""通用 PDF -> Markdown：anydoc 出正文，OvisOCR2 补图表。

思路
----
anydoc（Rust，毫秒级）读 PDF 的文本层与字体元数据，正文、斜体、超链接最完整；
但它把表格压成平铺文本，且完全不给图。OvisOCR2 相反：能还原 HTML 表格、
报出图的位置，代价是每页约 20 秒。

所以：先用 anydoc 转全文，再用 pymupdf 找出含图表的那几页，只对这些页跑
OvisOCR2，最后按锚点把补充内容插回 anydoc 输出的对应位置。

用法::

    python src/pdf2md.py paper.pdf -o out/
    python src/pdf2md.py paper.pdf -o out/ --pages 6,12
    python src/pdf2md.py paper.pdf -o out/ --all-pages
    python src/pdf2md.py scan.pdf  -o out/ --no-anydoc
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import time
from pathlib import Path

import pymupdf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from ovisocr2_llama import (  # noqa: E402
    BBOX_RE,
    DEFAULT_PROMPT,
    find_llama_cli,
    log,
    looks_corrupted,
    run_llama,
)

RENDER_DPI = 190

# Office 格式能走 anydoc 的结构化通道；PDF 不能（anydoc 只给 Markdown）
OFFICE_SUFFIXES = {".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
                   ".odt", ".ods", ".odp", ".rtf", ".epub", ".csv"}


def norm(s: str) -> str:
    """归一化文本用于模糊匹配：压掉空白。"""
    return re.sub(r"\s+", " ", s).strip()


def page_marker(page_no: int) -> str:
    return f"\n\n<!-- 第 {page_no} 页：OvisOCR2 补充 -->\n\n"


# ---------------------------------------------------------------- Office 结构导出

# ---------------------------------------------------------------- Office 结构导出

class _AssetWriter:
    """把 anydoc 的图片 asset 落盘，并返回 Markdown 引用。"""

    def __init__(self, doc, assets_dir: Path):
        self.by_id = {a.id: a for a in doc.assets}
        self.dir = assets_dir
        self.written: list[str] = []

    def ref(self, inline) -> str:
        src = getattr(inline, "source", None)
        aid = getattr(src, "asset_id", None) if src else None
        asset = self.by_id.get(aid)
        if asset is None:
            return ""
        ext = (asset.media_type or "image/png").split("/")[-1]
        ext = {"jpeg": "jpg"}.get(ext, ext)
        fname = f"img_{len(self.written) + 1}.{ext}"
        (self.dir / fname).write_bytes(asset.data)
        self.written.append(fname)
        alt = (getattr(inline, "alt", "") or "").strip()
        return f"![{alt}]({self.dir.name}/{fname})"


def _block_text(blk, writer: _AssetWriter) -> str:
    """取一个 block 的文本；inline 里的图片就地替换成 Markdown 图片引用。

    注意：单元格和列表项的正文都不在 `block.text`，而在 `block.content`
    的 Inline 列表里，所以要逐条拼接。
    """
    if getattr(blk, "content", None):
        out = []
        for item in blk.content:
            kind = getattr(item, "kind", None)
            if kind == "image":
                ref = writer.ref(item)
                if ref:
                    out.append(f"\n\n{ref}\n\n")
            elif kind == "text":
                out.append(getattr(item, "text", "") or "")
        joined = "".join(out)
        if joined.strip():
            return joined
    return getattr(blk, "text", None) or ""


def _slots_text(cell, writer: _AssetWriter) -> str:
    """取一个表格 Cell 的文本（内容在其 blocks 的 content 里）。"""
    parts = []
    for blk in getattr(cell, "blocks", None) or []:
        t = _block_text(blk, writer).strip()
        if t:
            parts.append(t)
    return " ".join(parts)


def _render_table(table, writer: _AssetWriter) -> str:
    """把 anydoc 的表格网格渲染成 HTML table。

    每个逻辑位置一个 CellSlot：`origin` 持有内容与 span，被覆盖的位置是
    `covered`（指向 origin），渲染时跳过，避免重复输出合并单元格。
    """
    grid = table.grid or []
    if not grid:
        return ""
    out = ['<table border="1">']
    for row in grid:
        out.append("<tr>")
        for slot in row:
            if slot is None:
                out.append("<td></td>")
                continue
            if getattr(slot, "kind", None) == "covered":
                continue                      # 已被 origin 的 span 覆盖
            cell = getattr(slot, "cell", None)
            if cell is None:
                out.append("<td></td>")
                continue
            span = ""
            rs = getattr(cell, "row_span", None)
            cs = getattr(cell, "col_span", None)
            if rs and rs > 1:
                span += f' rowspan="{rs}"'
            if cs and cs > 1:
                span += f' colspan="{cs}"'
            out.append(f"<td{span}>{_slots_text(cell, writer)}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "\n".join(out)


def office_to_markdown(doc, assets_dir: Path, stem: str) -> tuple[str, list[str]]:
    """把 anydoc Document 渲染成 Markdown：标题/段落/列表，表格转 HTML，图片落盘。"""
    writer = _AssetWriter(doc, assets_dir)
    lines: list[str] = []

    def render(blocks, list_depth: int = 0) -> None:
        for b in blocks:
            kind = getattr(b, "kind", None)

            if kind == "heading":
                lvl = getattr(b, "level", None) or 1
                t = _block_text(b, writer).strip()
                if t:
                    lines.append(f"{'#' * min(int(lvl), 6)} {t}\n")

            elif kind == "table" and getattr(b, "table", None) is not None:
                html = _render_table(b.table, writer)
                if html:
                    lines.append(html + "\n")

            elif kind == "list" and getattr(b, "list", None) is not None:
                indent = "  " * list_depth
                for item in b.list.items or []:
                    for ib in getattr(item, "blocks", None) or []:
                        t = _block_text(ib, writer).strip()
                        if t:
                            lines.append(f"{indent}- {t}")
                lines.append("")

            else:
                t = _block_text(b, writer).strip()
                if t:
                    lines.append(t + "\n")

            if kind == "list":
                continue                       # 列表项已单独处理

            if getattr(b, "blocks", None):
                render(b.blocks, list_depth)

    render(doc.blocks)
    return "\n".join(lines), writer.written


def detect_pages(doc, vector_threshold: int) -> list[int]:
    """找含图/表的页：位图、pymupdf 判定的表格、或矢量绘图超阈值。

    矢量绘图是关键信号——Tree Borrows 的流程图就是矢量画的，
    页面上一个位图对象都没有。
    """
    picked = []
    for i, page in enumerate(doc):
        n_img = len(page.get_images(full=True))
        n_draw = len(page.get_drawings())
        try:
            n_tab = len(page.find_tables().tables)
        except Exception:
            n_tab = 0
        if n_img > 0 or n_tab > 0 or n_draw >= vector_threshold:
            picked.append(i + 1)
    return picked


def page_anchor(page) -> str:
    """取该页最有辨识度的一段文字，用于在 anydoc 输出里定位。"""
    blocks = sorted(page.get_text("blocks"), key=lambda b: -len(b[4]))
    for b in blocks:
        t = norm(b[4])
        t = re.sub(r"^\d+:\d+\s*", "", t)
        t = re.sub(r"^\d+\s+", "", t)
        if len(t) > 80:
            return t[:80]
    return ""


def locate(haystack_norm: str, anchor: str) -> tuple[int, float]:
    """在归一化正文里找锚点，返回 (位置, 相似度)。"""
    if not anchor:
        return -1, 0.0
    probe = anchor[:40]
    best_pos, best_ratio = -1, 0.0
    for m in re.finditer(re.escape(probe), haystack_norm):
        seg = haystack_norm[m.start():m.start() + len(anchor)]
        ratio = difflib.SequenceMatcher(None, anchor, seg).ratio()
        if ratio > best_ratio:
            best_pos, best_ratio = m.start(), ratio
    return best_pos, best_ratio


def render_page(doc, page_no: int, out_png: Path, dpi: int) -> Image.Image:
    doc[page_no - 1].get_pixmap(dpi=dpi).save(out_png)
    return Image.open(out_png).convert("RGB")


def ocr_page(llama_bin, model, mmproj, png: Path, page: Image.Image,
             assets_dir: Path, max_tokens: int, ctx: int,
             threads: int, ngl: int) -> tuple[str, list[str]]:
    """跑一页 OCR，返回 (原始 markdown, 导出的图片文件名列表)。"""
    raw = run_llama(
        llama_bin=llama_bin, model=model, mmproj=mmproj, image_path=png,
        prompt=DEFAULT_PROMPT, max_tokens=max_tokens, ctx=ctx,
        threads=threads, ngl=ngl,
    )
    if looks_corrupted(raw):
        return "", []
    assets: list[str] = []
    w, h = page.size
    for left, top, right, bottom in BBOX_RE.findall(raw):
        x1 = max(0, round(int(left) * w / 1000))
        y1 = max(0, round(int(top) * h / 1000))
        x2 = min(w, round(int(right) * w / 1000))
        y2 = min(h, round(int(bottom) * h / 1000))
        if x2 <= x1 or y2 <= y1:
            continue
        fname = f"bbox_{left}_{top}_{right}_{bottom}.jpg"
        page.crop((x1, y1, x2, y2)).convert("RGB").save(assets_dir / fname)
        assets.append(fname)
    return raw, assets


def extract_supplement(raw: str, page_no: int, assets_dir: Path,
                       assets: list[str]) -> str:
    """只挑要补的部分：HTML 表格 + 图引用。正文文字不用 OvisOCR2 的。"""
    parts = [m.group(0) for m in
             re.finditer(r"<table.*?</table>", raw, re.DOTALL | re.IGNORECASE)]
    for fname in assets:
        parts.append(f'<img src="{assets_dir.name}/{fname}" />')
    if not parts:
        return ""
    return page_marker(page_no) + "\n\n".join(parts) + "\n"


def insert_at_norm_pos(text: str, norm_pos: int, payload: str) -> str:
    """把 payload 插到 norm() 坐标系第 norm_pos 个字符对应的原文位置之前。

    注意坐标要跟 norm() 一致：连续空白折叠成**一个空格**，空格本身也占一位。
    只数非空白字符会让位置不断前移，最终插进单词中间。
    """
    if norm_pos < 0:
        return text + payload
    n = len(text)
    i = 0
    while i < n and text[i].isspace():      # 对应 norm() 的 strip()
        i += 1
    count = 0
    while i < n and count < norm_pos:
        if text[i].isspace():
            while i < n and text[i].isspace():   # 折叠成一个空格
                i += 1
            count += 1
        else:
            i += 1
            count += 1
    return text[:i] + payload + text[i:]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="PDF -> Markdown：anydoc 出正文，OvisOCR2 补图表")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=Path("out"))
    ap.add_argument("--no-anydoc", action="store_true",
                    help="跳过 anydoc（扫描件用，正文也交给 OvisOCR2）")
    ap.add_argument("--no-ocr", action="store_true", help="跳过 OvisOCR2，纯 anydoc")
    ap.add_argument("--all-pages", action="store_true", help="所有页都跑 OCR")
    ap.add_argument("--pages", default=None, help="手动指定页，如 6,12,16")
    ap.add_argument("--vector-threshold", type=int, default=15,
                    help="矢量绘图数达到该值即视为含图（默认 15）")
    ap.add_argument("--dpi", type=int, default=RENDER_DPI)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("-c", "--ctx-size", type=int, default=16384)
    ap.add_argument("-t", "--threads", type=int, default=8)
    ap.add_argument("-ngl", "--ngl", type=int, default=99)
    ap.add_argument("-m", "--model", default="models/OvisOCR2-Q4_K_M.gguf")
    ap.add_argument("--mmproj", default="models/mmproj-F16.gguf")
    ap.add_argument("--llama-bin", default=None)
    args = ap.parse_args(argv)

    if not args.pdf.exists():
        raise SystemExit(f"PDF 不存在: {args.pdf}")
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.pdf.stem
    assets_dir = args.output / f"{stem}.assets"

    log(f"输入: {args.pdf.name}")

    # ---- 0. Office 格式：走 anydoc 结构化通道，不渲染也不 OCR
    if args.pdf.suffix.lower() in OFFICE_SUFFIXES:
        try:
            import anydoc
        except ImportError:
            raise SystemExit("处理 Office 格式需要 anydoc：pip install firecrawl-anydoc")
        t0 = time.perf_counter()
        odoc = anydoc.to_document(args.pdf.read_bytes())
        assets_dir.mkdir(parents=True, exist_ok=True)
        md, written = office_to_markdown(odoc, assets_dir, stem)
        dt = (time.perf_counter() - t0) * 1000
        out_md = args.output / f"{stem}.md"
        out_md.write_text(md, encoding="utf-8")
        log(f"anydoc 结构化导出: {dt:.0f} ms, {len(md)} 字符, "
            f"{len(written)} 张图（无损，无需 OCR）")
        log(f"输出 {out_md}")
        return 0

    doc = pymupdf.open(args.pdf)
    log(f"PDF: {doc.page_count} 页")

    # ---- 1. 正文（PDF）
    if args.no_anydoc:
        base_md = ""
        log("跳过 anydoc，正文将由 OvisOCR2 提供")
    else:
        try:
            import anydoc
            t0 = time.perf_counter()
            base_md = anydoc.to_markdown(str(args.pdf))
            log(f"anydoc: {len(base_md)} 字符，{(time.perf_counter()-t0)*1000:.0f} ms")
        except ImportError:
            log("未装 anydoc（pip install firecrawl-anydoc），回退 pymupdf 文本层")
            base_md = "\n\n".join(p.get_text() for p in doc)
        except Exception as e:
            log(f"anydoc 失败（{type(e).__name__}: {e}），回退 pymupdf 文本层")
            base_md = "\n\n".join(p.get_text() for p in doc)

    # ---- 2. 选页
    if args.pages:
        targets = [int(x) for x in args.pages.replace(" ", "").split(",") if x]
    elif args.no_ocr:
        targets = []
    elif args.all_pages or args.no_anydoc:
        targets = list(range(1, doc.page_count + 1))
    else:
        targets = detect_pages(doc, args.vector_threshold)
    log(f"OCR 目标页: {targets or '（无）'}  {len(targets)}/{doc.page_count}")

    # ---- 3. 逐页 OCR
    supplements: list[tuple[int, str]] = []
    if targets:
        llama_bin = find_llama_cli(args.llama_bin)
        assets_dir.mkdir(parents=True, exist_ok=True)
        tmp_png = args.output / f".{stem}.page.png"
        for n in targets:
            log(f"OCR p{n}")
            page_img = render_page(doc, n, tmp_png, args.dpi)
            raw, assets = ocr_page(
                llama_bin, args.model, args.mmproj, tmp_png, page_img,
                assets_dir, args.max_new_tokens, args.ctx_size,
                args.threads, args.ngl)
            if not raw:
                log("  该页输出损坏，跳过")
                continue
            sup = extract_supplement(raw, n, assets_dir, assets)
            if sup:
                supplements.append((n, sup))
                log(f"  补入 {len(assets)} 图 / {sup.count('<table')} 表格")
            else:
                log("  该页无表格与图，无需补充")
        tmp_png.unlink(missing_ok=True)

    # ---- 4. 插回正文
    final = base_md
    base_norm = norm(base_md)
    fallback: list[str] = []

    if args.no_anydoc:
        # 全程 OCR：直接按页拼接
        chunks = []
        for page_no, sup in supplements:
            chunks.append(sup)
        final = "".join(chunks)
    else:
        for page_no, sup in supplements:
            anchor = page_anchor(doc[page_no - 1])
            pos, ratio = locate(base_norm, anchor)
            if pos >= 0 and ratio > 0.6:
                final = insert_at_norm_pos(final, pos, sup)
                log(f"p{page_no} 插入正文（匹配 {ratio:.2f}）")
            else:
                fallback.append(sup)
                log(f"p{page_no} 定位失败（{ratio:.2f}），放文末附录")
        if fallback:
            final += "\n\n---\n\n# 附：图表页补充\n" + "\n".join(fallback)

    out_md = args.output / f"{stem}.md"
    out_md.write_text(final, encoding="utf-8")
    log(f"输出 {out_md}（{len(final)} 字符）")
    if assets_dir.exists():
        n_jpg = len(list(assets_dir.glob("*.jpg")))
        if n_jpg:
            log(f"图片 {assets_dir}/（{n_jpg} 张）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
