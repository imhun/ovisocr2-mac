#!/usr/bin/env bash
#
# OvisOCR2 一键构建：环境检查 -> Python 依赖 -> GGUF 权重 -> llama.cpp（委托给
# 独立仓库去 clone、打补丁、编译、自检）-> 冒烟测试。
#
# 设计成可以反复执行：每一步都会先检查是否已经完成，完成了就跳过。
# 想从头再来加 --clean。
#
#   ./scripts/build.sh                 # 完整流程
#   ./scripts/build.sh --skip-models   # 权重已下好
#   ./scripts/build.sh --clean         # 删掉工作目录重建
#   ./scripts/build.sh --help

set -euo pipefail

# ---------------------------------------------------------------- 基本路径

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p tmp

# llama.cpp 的构建在独立仓库里维护（submodule 挂在 third_party/ 下）：
# 补丁、pin 的版本、编译开关都在那边，这里只调用它。
LLAMA_REPO_DIR="third_party/llamacpp-metal-amd"
LLAMA_WORK_DIR="tmp/llama.cpp"

JOBS="$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
SKIP_MODELS=0
SKIP_DEPS=0
SKIP_SMOKE=0
SKIP_BUILD=0
DO_CLEAN=0

MODEL="models/OvisOCR2-Q4_K_M.gguf"
MMPROJ="models/mmproj-F16.gguf"
MODEL_BYTES=529297280
MMPROJ_BYTES=204987079

# ---------------------------------------------------------------- 输出

if [ -t 1 ]; then
    BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'
else
    BOLD=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi

step()  { printf '\n%s==> %s%s\n' "$BOLD" "$*" "$RESET"; }
ok()    { printf '%s  ✓%s %s\n' "$GREEN" "$RESET" "$*"; }
warn()  { printf '%s  !%s %s\n' "$YELLOW" "$RESET" "$*"; }
die()   { printf '%s  ✗ %s%s\n' "$RED" "$*" "$RESET" >&2; exit 1; }

usage() {
    cat <<'EOF'
OvisOCR2 一键构建

  环境检查 -> Python 依赖 -> GGUF 权重
  -> llama.cpp（调 third_party/llamacpp-metal-amd 完成 clone / 打补丁 / 编译 / 自检）
  -> 冒烟测试

每一步都会先检查是否已完成，可以反复执行；想从头再来加 --clean。

用法:
  ./scripts/build.sh [选项]

选项:
  --work-dir DIR    llama.cpp 工作目录（默认 tmp/llama.cpp）
  -j, --jobs N      并行编译数（默认 CPU 核数）
  --skip-models     跳过权重下载
  --skip-deps       跳过 Python 依赖安装
  --skip-build      只准备 llama.cpp 源码与补丁，不编译
  --skip-smoke      跳过冒烟测试
  --clean           删掉工作目录后重新 clone 并打补丁
  -h, --help        显示这段帮助

国内网络下载权重时指定镜像:
  HF_BASE=https://hf-mirror.com/Abiray/OvisOCR2-GGUF/resolve/main ./scripts/build.sh
EOF
    exit 0
}

# ---------------------------------------------------------------- 参数

while [ $# -gt 0 ]; do
    case "$1" in
        --work-dir)   LLAMA_WORK_DIR="$2"; shift 2 ;;
        --jobs|-j)    JOBS="$2"; shift 2 ;;
        --skip-models) SKIP_MODELS=1; shift ;;
        --skip-deps)  SKIP_DEPS=1; shift ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        --skip-smoke) SKIP_SMOKE=1; shift ;;
        --clean)      DO_CLEAN=1; shift ;;
        -h|--help)    usage ;;
        *)            die "未知参数：$1（用 --help 看用法）" ;;
    esac
done

LLAMA_BIN="$REPO_ROOT/$LLAMA_WORK_DIR/build-metal/bin/llama-mtmd-cli"

# ---------------------------------------------------------------- 0. 环境检查

step "0/4 环境检查"

for tool in git cmake curl; do
    command -v "$tool" >/dev/null 2>&1 || die "缺少 $tool，请先安装"
done
ok "git / cmake / curl 都在"

if [ "$(uname -s)" != "Darwin" ]; then
    die "这个脚本针对 macOS；其他平台直接按 README 编译 llama.cpp 即可"
fi
xcode-select -p >/dev/null 2>&1 || die "缺少 Xcode 命令行工具，先跑 xcode-select --install"
ok "Xcode 命令行工具：$(xcode-select -p)"

ARCH="$(uname -m)"
case "$ARCH" in
    x86_64) ok "架构 x86_64（Intel，走 README 里的 SIMD 开关）" ;;
    arm64)  ok "架构 arm64（Apple Silicon，不启用 x86 SIMD 开关）" ;;
    *)      die "不认识的架构：$ARCH" ;;
esac

if ! command -v uv >/dev/null 2>&1; then
    warn "没有 uv，Python 依赖会退回 pip 安装"
    UV=0
else
    UV=1
    ok "uv $(uv --version | awk '{print $2}')"
fi

if [ "$DO_CLEAN" -eq 1 ]; then
    step "清理 $WORK_DIR"
    [ -e "$LLAMA_WORK_DIR" ] && mv "$LLAMA_WORK_DIR" "$LLAMA_WORK_DIR.removed-$(date +%s)"
    ok "已删除，接下来会重新 clone 并打补丁"
fi

# ---------------------------------------------------------------- 1. Python 依赖

step "1/4 Python 环境"

if [ "$SKIP_DEPS" -eq 1 ]; then
    warn "--skip-deps，跳过"
else
    if [ "$UV" -eq 1 ]; then
        [ -d .venv ] || uv venv --python 3.12 .venv
        uv pip install -q -r requirements.txt
        ok "依赖已装到 .venv（$(.venv/bin/python -c 'import sys; print(sys.version.split()[0])')）"
    else
        [ -d .venv ] || python3 -m venv .venv
        .venv/bin/python -m pip install -q --upgrade pip
        .venv/bin/python -m pip install -q -r requirements.txt
        ok "依赖已装到 .venv"
    fi
fi

# ---------------------------------------------------------------- 2. 权重

step "2/4 GGUF 权重"

size_of() { [ -f "$1" ] && wc -c < "$1" | tr -d ' ' || echo 0; }

if [ "$SKIP_MODELS" -eq 1 ]; then
    warn "--skip-models，跳过"
elif [ "$(size_of "$MODEL")" -ge "$MODEL_BYTES" ] && [ "$(size_of "$MMPROJ")" -ge "$MMPROJ_BYTES" ]; then
    ok "权重齐全（已存在且大小达标）"
else
    warn "权重不全或缺失，开始下载（断点续传，可中断后重跑）"
    echo "     国内网络建议先指定镜像："
    echo "     HF_BASE=https://hf-mirror.com/Abiray/OvisOCR2-GGUF/resolve/main ./scripts/build.sh"
    sh scripts/fetch_gguf.sh
    ok "权重下载完成"
fi

# ---------------------------------------------------------------- 3. llama.cpp
#
# clone、打补丁、编译、自检都在独立仓库里做，这里只负责用本仓库的 tmp/ 当工作目录。
# 补丁不需要在本仓库再存一份。

step "3/4 llama.cpp（委托 $LLAMA_REPO_DIR）"

if [ ! -d "$LLAMA_REPO_DIR" ] || [ -z "$(ls -A "$LLAMA_REPO_DIR" 2>/dev/null)" ]; then
    warn "子模块不在，先拉取"
    git submodule update --init --recursive "$LLAMA_REPO_DIR" \
        || die "子模块拉取失败。GitHub 直连不通时先设代理：
     export https_proxy=http://127.0.0.1:7897 http_proxy=http://127.0.0.1:7897"
fi
[ -x "$LLAMA_REPO_DIR/scripts/build.sh" ] \
    || die "$LLAMA_REPO_DIR 内容不完整（缺 scripts/build.sh）"

mkdir -p "$LLAMA_WORK_DIR"

SUBBUILD_ARGS=(--work-dir "$REPO_ROOT/$LLAMA_WORK_DIR" -j "$JOBS")
if [ "$SKIP_BUILD" -eq 1 ]; then
    SUBBUILD_ARGS+=(--patches-only)
fi
if [ "$DO_CLEAN" -eq 1 ]; then
    SUBBUILD_ARGS+=(--clean)
fi

"$LLAMA_REPO_DIR/scripts/build.sh" "${SUBBUILD_ARGS[@]}" || die "llama.cpp 构建失败"

if [ "$SKIP_BUILD" -eq 1 ]; then
    warn "--skip-build，只准备了源码与补丁"
else
    [ -x "$LLAMA_BIN" ] || die "构建结束但找不到 $LLAMA_BIN"
    ok "llama-mtmd-cli: ${LLAMA_BIN#$REPO_ROOT/}"
fi

# ---------------------------------------------------------------- 4. 冒烟测试

step "4/4 冒烟测试"

if [ "$SKIP_SMOKE" -eq 1 ]; then
    warn "--skip-smoke，跳过"
elif [ "$(size_of "$MODEL")" -lt "$MODEL_BYTES" ] || [ "$(size_of "$MMPROJ")" -lt "$MMPROJ_BYTES" ]; then
    warn "权重不齐，跳过冒烟测试"
elif [ ! -x .venv/bin/python ]; then
    warn "没有 .venv，跳过冒烟测试"
else
    SMOKE_DIR="tmp/smoke"
    mkdir -p "$SMOKE_DIR"
    .venv/bin/python - "$SMOKE_DIR/page.png" <<'PY'
import sys
from PIL import Image, ImageDraw

img = Image.new("RGB", (1000, 500), "white")
d = ImageDraw.Draw(img)
d.text((60, 80), "OvisOCR2 smoke test", fill="black")
d.text((60, 160), "build 20260101 OK", fill="black")
d.rectangle((60, 240, 600, 420), outline="black", width=3)
d.line((60, 330, 600, 330), fill="black", width=2)
d.line((330, 240, 330, 420), fill="black", width=2)
d.text((80, 270), "1", fill="black")
d.text((350, 270), "2", fill="black")
img.save(sys.argv[1])
PY

    set +e
    .venv/bin/python src/ovisocr2_llama.py "$SMOKE_DIR/page.png" -o "$SMOKE_DIR/out" \
        --llama-bin "$LLAMA_BIN" --verbose > "$SMOKE_DIR/run.log" 2>&1
    SMOKE_RC=$?
    set -e

    [ "$SMOKE_RC" -eq 0 ] || die "冒烟测试失败（退出码 $SMOKE_RC），日志：$SMOKE_DIR/run.log"

    SMOKE_MD="$SMOKE_DIR/out/page.md"
    [ -s "$SMOKE_MD" ] || die "冒烟测试没有输出 Markdown"

    if grep -q '@@@@' "$SMOKE_MD"; then
        die "输出出现 @@@@，GPU 后端不稳（可能是分辨率过高触发 watchdog）"
    fi

    if ! grep -qi "smoke" "$SMOKE_MD"; then
        warn "输出里没找到 'smoke' 字样，看一下 $SMOKE_MD 确认结果"
    else
        ok "OCR 输出正常：$(tr -d '\n' < "$SMOKE_MD" | cut -c1-60)..."
    fi

    if grep -q "ggml_metal: device 0" "$SMOKE_DIR/run.log"; then
        GPU_LINE=$(grep -m1 "ggml_metal: device 0" "$SMOKE_DIR/run.log" | sed 's/^.*device 0: //')
        ok "Metal 已启用：$GPU_LINE"
    else
        warn "日志里没有 Metal 设备信息，可能退回了 CPU"
    fi
fi

# ---------------------------------------------------------------- 收尾

step "完成"
cat <<EOF
  推理（llama.cpp + Metal）:  .venv/bin/python src/ovisocr2_llama.py page.jpg -o out
  文档转 Markdown          :  .venv/bin/python src/pdf2md.py paper.pdf -o out/

  默认长边 1600，超过 2000 会触发 macOS GPU watchdog，详见 README 第五节。
EOF
