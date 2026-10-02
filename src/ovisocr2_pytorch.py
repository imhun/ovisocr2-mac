"""OvisOCR2 文档解析推理（PyTorch / MPS）。

用法::

    python src/ovisocr2_pytorch.py page.jpg -o out/page.md
    python src/ovisocr2_pytorch.py scans/ -o out/ --device mps

设备策略：mps > cuda > cpu。MPS 不支持 bfloat16，脚本会自动把 bf16 权重
降为 float16（OvisOCR2 官方权重即为 bf16，Apple Silicon 上同样需要转换）。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image

MODEL_ID = "ATH-MaaS/OvisOCR2"

# 官方 prompt（见 HF 模型卡 Inference 一节）
PROMPT = (
    "Extract all readable content from the image in natural human reading order "
    "and output the result as a single Markdown document. For charts or images, "
    'represent them using an HTML image tag: <img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, '
    "where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). "
    "Format formulas as LaTeX. Format tables as HTML: <table>...</table>. "
    "Transcribe all other text as standard Markdown. "
    "Preserve the original text without translation or paraphrasing."
)

BBOX_RE = re.compile(r'<img src="images/bbox_(\d+)_(\d+)_(\d+)_(\d+)\.jpg" />')

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def log(msg: str) -> None:
    print(f"[ovisocr2] {msg}", flush=True)


def pick_device(requested: str | None) -> torch.device:
    """按 mps > cuda > cpu 的顺序挑选设备；requested 非空时强制使用。"""
    if requested:
        dev = torch.device(requested)
        if dev.type == "mps" and not torch.backends.mps.is_available():
            log("警告：显式指定 mps 但当前不可用，回退 cpu")
            return torch.device("cpu")
        return dev
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    """MPS 不支持 bf16，统一在 MPS 上用 fp16；CPU 上用 fp32 保证精度。"""
    if device.type == "mps":
        return torch.float16
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def clean_truncated_repeats(
    text: str,
    min_text_len: int = 8000,
    max_period: int = 200,
    min_period: int = 1,
    min_repeat_chars: int = 100,
    min_repeat_times: int = 5,
) -> str:
    """官方给出的尾部重复输出清理逻辑，原样移植。"""
    n = len(text)
    if n < min_text_len:
        return text
    max_period = min(max_period, n - 1)
    for unit_len in range(min_period, max_period + 1):
        if text[n - 1] != text[n - 1 - unit_len]:
            continue
        match_len = 1
        idx = n - 2
        while idx >= unit_len and text[idx] == text[idx - unit_len]:
            match_len += 1
            idx -= 1
        total_len = match_len + unit_len
        repeat_times = total_len // unit_len
        tail_len = total_len % unit_len
        if repeat_times >= min_repeat_times and total_len >= min_repeat_chars:
            return text[: n - total_len + unit_len] + text[n - tail_len :]
    return text


class OvisOCR2Parser:
    """薄封装：加载一次模型，重复解析多张页面图。"""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
        max_pixels: int = 2880 * 2880,
        min_pixels: int = 448 * 448,
    ) -> None:
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        self.device = pick_device(device)
        self.dtype = pick_dtype(self.device)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        log(f"设备={self.device} dtype={str(self.dtype).replace('torch.', '')}")
        log(f"加载 processor: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)

        log(f"加载模型权重: {model_id}（首次运行需下载 ~1.6GB）")
        t0 = time.perf_counter()
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        self.model = self.model.to(self.device).eval()
        log(f"模型就绪，用时 {time.perf_counter() - t0:.1f}s")

        self.prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    @torch.no_grad()
    def parse(self, image: Image.Image, max_new_tokens: int = 16384) -> str:
        inputs = self.processor(
            text=[self.prompt],
            images=[image],
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        t0 = time.perf_counter()
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if self.device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - t0

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = out.shape[1] - prompt_len
        rate = new_tokens / max(elapsed, 1e-6)
        log(f"生成 {new_tokens} tokens，用时 {elapsed:.1f}s ({rate:.1f} tok/s)")

        text = self.processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True)[0].strip()
        return clean_truncated_repeats(text)


def strip_bbox_tags(markdown: str) -> str:
    """去掉视觉区域的 <img ... /> 占位（等价官方 filter_imgtags=True）。"""
    blocks = [
        b for b in markdown.split("\n\n")
        if not b.strip().startswith('<img src="images/bbox_')
    ]
    return "\n\n".join(blocks)


def save_with_visual_regions(markdown: str, page: Image.Image, out_dir: Path) -> None:
    """把 <img> 占位替换成真实页面裁剪图（等价官方 filter_imgtags=False）。"""
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    width, height = page.size
    for left, top, right, bottom in BBOX_RE.findall(markdown):
        x1 = max(0, min(width, round(int(left) * width / 1000)))
        y1 = max(0, min(height, round(int(top) * height / 1000)))
        x2 = max(0, min(width, round(int(right) * width / 1000)))
        y2 = max(0, min(height, round(int(bottom) * height / 1000)))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = page.crop((x1, y1, x2, y2)).convert("RGB")
        crop.save(images_dir / f"bbox_{left}_{top}_{right}_{bottom}.jpg")


def collect_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not path.exists():
        raise SystemExit(f"输入不存在: {path}")
    return [path]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="OvisOCR2 文档解析（PyTorch / MPS）")
    ap.add_argument("input", type=Path, help="页面图片，或包含图片的目录")
    ap.add_argument("-o", "--output", type=Path, default=Path("out"), help="输出目录")
    ap.add_argument("--device", default=None, help="强制设备：mps / cuda / cpu")
    ap.add_argument("--model", default=MODEL_ID, help="模型 ID 或本地路径")
    ap.add_argument("--max-new-tokens", type=int, default=16384)
    ap.add_argument(
        "--keep-visual-regions",
        action="store_true",
        help="保留 <img> 占位并导出对应裁剪图（默认去掉）",
    )
    args = ap.parse_args(argv)

    # MPS 遇到未实现算子时回退 CPU，而不是直接崩溃
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    pages = collect_inputs(args.input)
    if not pages:
        raise SystemExit(f"目录中没有图片: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    parser = OvisOCR2Parser(model_id=args.model, device=args.device)

    for page_path in pages:
        log(f"解析 {page_path.name}")
        page = Image.open(page_path).convert("RGB")
        markdown = parser.parse(page, max_new_tokens=args.max_new_tokens)

        text = markdown if args.keep_visual_regions else strip_bbox_tags(markdown)

        md_path = args.output / f"{page_path.stem}.md"
        md_path.write_text(text, encoding="utf-8")
        if args.keep_visual_regions:
            save_with_visual_regions(markdown, page, args.output / page_path.stem)
        log(f"已写出 {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
