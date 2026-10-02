"""OvisOCR2 文档解析（llama.cpp / GGUF）。

首选后端：ToshLLM 补丁版的 Metal（AMD GPU 加速）。
  AMD Radeon Pro 5500M 实测：视觉编码 6.2s，整页 15s，输出正确。
  对比 stock llama.cpp：Metal 触发 GPU Timeout 输出乱码；Vulkan 的文本路径本身没问题，
  但视觉编码器算错（误差随图像分辨率放大），CPU 则要 159s 编码。归因与实测数据见
  llamacpp-metal-amd 仓库的 README。

回退后端：官方 llama.cpp 的 CPU 路径（Accelerate）。
  此时建议用 --max-side 1000 缩图，可把视觉编码从 159s 降到 26s。

Vulkan 后端（可选）：文本放 AMD 独显，视觉编码回退 CPU，输出与 Metal 逐字节一致。
  必须同时给 --device Vulkan0 和 --no-mmproj-offload：前者防止层被拆到 Metal/Vulkan
  两个后端上（那样是整屏 @@@@），后者避开 Vulkan 视觉编码的数值错误（否则退化成
  "1 1 1…"）::

    python src/ovisocr2_llama.py page.jpg -o out \
        --llama-bin tmp/llama.cpp/build-vulkan/bin/llama-mtmd-cli \
        --device Vulkan0 --no-mmproj-offload

用法::

    python src/ovisocr2_llama.py page.jpg -o out
    python src/ovisocr2_llama.py scans/ -o out                  # ToshLLM Metal
    python src/ovisocr2_llama.py scans/ -o out --max-side 1000  # CPU 回退时缩图
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

DEFAULT_PROMPT = (
    "Extract all readable content from the image in natural human reading order "
    "and output the result as a single Markdown document. For charts or images, "
    'represent them using an HTML image tag: <img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, '
    "where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). "
    "Format formulas as LaTeX. Format tables as HTML: <table>...</table>. "
    "Transcribe all other text as standard Markdown. "
    "Preserve the original text without translation or paraphrasing."
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# OvisOCR2 用这个占位符标注图表/流程图等视觉区域，坐标已归一化到 [0,1000)
BBOX_RE = re.compile(r'<img src="images/bbox_(\d+)_(\d+)_(\d+)_(\d+)\.jpg" />')

# 实测：长边超过此值后 AMD GPU 的视觉编码 kernel 会触发 macOS watchdog
# （kIOAccelCommandBufferCallbackErrorTimeout），输出退化成 "## 1" 之类乱码。
# 1600 编码约 3s，2000 约 7.6s，2400 起开始出现 timeout，2800 起必坏。
GPU_SAFE_MAX_SIDE = 2000
DEFAULT_MAX_SIDE = 1600

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def log(msg: str) -> None:
    print(f"[ovisocr2] {msg}", flush=True)


def strip_think(text: str) -> str:
    """Qwen3.5 会输出空的  thinking 块，去掉后再作为 Markdown 保存。"""
    return THINK_RE.sub("", text).strip()


def looks_corrupted(markdown: str) -> bool:
    """识别 GPU timeout 造成的退化输出。

    实测坏掉时有两种形态：整屏 '@'，或每个标题都是重复的 "## 1"。
    """
    if "@@@@" in markdown:
        return True
    repeated = re.findall(r"^#{1,6}\s+1\s*$", markdown, re.MULTILINE)
    return len(repeated) >= 3


def strip_bbox_tags(markdown: str) -> str:
    """去掉图表/流程图的 <img> 占位（默认行为）。

    OvisOCR2 不会把流程图转成文字，只会标出它的位置；默认把这些占位删掉，
    需要保留时用 --keep-visual-regions。
    """
    blocks = [
        b for b in markdown.split("\n\n")
        if not b.strip().startswith('<img src="images/bbox_')
    ]
    return "\n\n".join(blocks)


def save_with_visual_regions(markdown: str, page: Image.Image, out_dir: Path) -> None:
    """按 <img> 占位里的 bbox 从原页裁出图片，供 Markdown 引用。"""
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


def find_llama_cli(explicit: str | None) -> str:
    if explicit:
        return explicit
    # 优先级：本仓库构建的 > 独立仓库默认路径 > 全局安装 > PATH > 官方源码构建
    candidates = [
        Path("tmp/llama.cpp/build-metal/bin/llama-mtmd-cli"),
        Path("third_party/llamacpp-metal-amd/tmp/llama.cpp/build-metal/bin/llama-mtmd-cli"),
        # ./scripts/install.sh 装到全局的位置（供多个项目共用）
        Path("/usr/local/lib/llamacpp-metal-amd/llama-mtmd-cli"),
        Path("/opt/homebrew/lib/llamacpp-metal-amd/llama-mtmd-cli"),
        # 早期手工构建留下的路径，保留兼容
        Path("tmp/tosh-llama/build-tosh/bin/llama-mtmd-cli"),
        Path("tmp/llama.cpp/build/bin/llama-mtmd-cli"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    # llama-mtmd-cli-amd 是 install.sh 装的补丁版；后面那个通用名字可能是
    # brew 装的官方版（在独显上输出是乱的），只当最后兜底。
    for name in ("llama-mtmd-cli-amd", "llama-mtmd-cli"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit(
        "找不到 llama-mtmd-cli。请先安装 llama.cpp（brew install llama.cpp），"
        "或用 --llama-bin 指定路径。"
    )


def shrink_page(image: Image.Image, max_side: int, workdir: Path, name: str) -> Path:
    """max_side <= 0 时保持原尺寸；否则把长边限制到 max_side。

    视觉编码耗时与分辨率成正比。GPU 后端足够快，可保持原图；
    CPU 后端建议 --max-side 1000。
    """
    if max_side <= 0:
        out = workdir / f"{name}.png"
        image.save(out)
        return out
    width, height = image.size
    longest = max(width, height)
    if max_side > GPU_SAFE_MAX_SIDE:
        log(f"警告：--max-side {max_side} 超过安全上限 {GPU_SAFE_MAX_SIDE}，"
            f"可能触发 GPU timeout 导致输出损坏，已按上限处理")
        max_side = GPU_SAFE_MAX_SIDE
    if longest > max_side:
        ratio = max_side / longest
        image = image.resize((max(1, int(width * ratio)), max(1, int(height * ratio))), Image.LANCZOS)
    out = workdir / f"{name}.png"
    image.save(out)
    return out


def run_llama(
    llama_bin: str,
    model: str,
    mmproj: str,
    image_path: Path,
    prompt: str,
    max_tokens: int,
    ctx: int,
    threads: int,
    ngl: int,
    temp: float = 0.0,
    seed: int = 42,
    device: str | None = None,
    mmproj_offload: bool = True,
    verbose: bool = False,
) -> str:
    cmd = [
        llama_bin,
        "-m", model,
        "--mmproj", mmproj,
        "--image", str(image_path),
        "-p", prompt,
        "-n", str(max_tokens),
        "-c", str(ctx),
        "-t", str(threads),
        "-ngl", str(ngl),
        "--no-warmup",
        # 文档解析要可复现：默认贪心解码，同一页重跑结果一致
        "--temp", str(temp),
        "--seed", str(seed),
    ]
    if device:
        # 多后端二进制（Metal + Vulkan 都在）不锁设备的话，层会被拆到两个后端上
        cmd += ["-dev", device]
    if not mmproj_offload:
        # Vulkan 上视觉编码在长边超过 ~1024 后会算错，放回 CPU 才是对的
        cmd.append("--no-mmproj-offload")
    # 日志走 stderr，stdout 只保留生成的 Markdown
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if verbose and proc.stderr:
        # --verbose 时把 llama.cpp 的日志透传出来，用来确认接管的是哪个后端
        # （ToshLLM Metal 会打印 "ggml_metal: device 0: ... not bridged"）
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"llama-mtmd-cli 失败 (code={proc.returncode}):\n{proc.stderr[-2000:]}")
    return strip_think(proc.stdout)


def collect_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not path.exists():
        raise SystemExit(f"输入不存在: {path}")
    return [path]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="OvisOCR2 文档解析（llama.cpp CPU 路径）")
    ap.add_argument("input", type=Path, help="页面图片，或包含图片的目录")
    ap.add_argument("-o", "--output", type=Path, default=Path("out"), help="输出目录")
    ap.add_argument("-m", "--model", default="models/OvisOCR2-Q4_K_M.gguf")
    ap.add_argument("--mmproj", default="models/mmproj-F16.gguf")
    ap.add_argument("--llama-bin", default=None, help="llama-mtmd-cli 路径")
    ap.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE,
                    help=f"图像长边上限（默认 {DEFAULT_MAX_SIDE}）；"
                         f"超过 {GPU_SAFE_MAX_SIDE} 会被压到安全值。CPU 后端建议 1000")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("-c", "--ctx-size", type=int, default=8192)
    ap.add_argument("-t", "--threads", type=int, default=8)
    ap.add_argument("-ngl", "--ngl", type=int, default=99,
                    help="offload 到 GPU 的层数；99=全部（无 GPU 后端时忽略）")
    ap.add_argument("--temp", type=float, default=0.0,
                    help="采样温度（默认 0 = 贪心解码，同一页重跑结果一致）")
    ap.add_argument("--seed", type=int, default=42,
                    help="采样随机种子（默认 42；温度为 0 时无影响）")
    ap.add_argument("--device", default=None,
                    help="把模型锁到单个设备上（透传 -dev），例如 Vulkan0。"
                         "同时带 Metal 和 Vulkan 的二进制不锁设备会把层拆开，输出会坏")
    ap.add_argument("--no-mmproj-offload", action="store_true",
                    help="视觉编码器不上 GPU（透传 --no-mmproj-offload）。"
                         "Vulkan 上视觉编码在长边超过 ~1024 后会算错，必须加这个")
    ap.add_argument("--keep-visual-regions", action="store_true",
                    help="保留 <img> 占位并导出对应裁剪图（图表/流程图）；"
                         "默认去掉这些占位")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="透传 llama.cpp 的日志到 stderr（确认 GPU 后端、排查损坏输出）")
    args = ap.parse_args(argv)

    llama_bin = find_llama_cli(args.llama_bin)
    log(f"llama-mtmd-cli: {llama_bin}")
    if args.device and "vulkan" in args.device.lower() and not args.no_mmproj_offload:
        log("警告：Vulkan 的视觉编码器在图像长边超过 ~1024 后会算错"
            "（实测 1600 时偏差 500%+），建议加上 --no-mmproj-offload 放回 CPU")

    pages = collect_inputs(args.input)
    if not pages:
        raise SystemExit(f"目录中没有图片: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=args.output) as tmp:
        tmpdir = Path(tmp)
        for page_path in pages:
            log(f"解析 {page_path.name}")
            page = Image.open(page_path).convert("RGB")
            prepared = shrink_page(page, args.max_side, tmpdir, page_path.stem)

            markdown = run_llama(
                llama_bin=llama_bin,
                model=args.model,
                mmproj=args.mmproj,
                image_path=prepared,
                prompt=DEFAULT_PROMPT,
                max_tokens=args.max_new_tokens,
                ctx=args.ctx_size,
                threads=args.threads,
                ngl=args.ngl,
                temp=args.temp,
                seed=args.seed,
                device=args.device,
                mmproj_offload=not args.no_mmproj_offload,
                verbose=args.verbose,
            )

            text = markdown if args.keep_visual_regions else strip_bbox_tags(markdown)

            md_path = args.output / f"{page_path.stem}.md"
            md_path.write_text(text, encoding="utf-8")
            if args.keep_visual_regions:
                save_with_visual_regions(markdown, page, args.output / page_path.stem)
            log(f"已写出 {md_path} ({len(text)} 字符)")
            if looks_corrupted(text):
                log(f"警告：{page_path.name} 的输出疑似被 GPU timeout 损坏，"
                    f"建议调小 --max-side 后重跑")

    return 0


if __name__ == "__main__":
    sys.exit(main())
