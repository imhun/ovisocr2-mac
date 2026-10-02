#!/bin/sh
# 带续传与重试的 GGUF 下载（网络不稳定时反复重试直到大小达标）
set -u

OUT_DIR="models"
# 国内网络直连 HF 常超时，用 HF_BASE 指向镜像，例如：
#   HF_BASE=https://hf-mirror.com/Abiray/OvisOCR2-GGUF/resolve/main ./scripts/fetch_gguf.sh
BASE="${HF_BASE:-https://huggingface.co/Abiray/OvisOCR2-GGUF/resolve/main}"

# 文件名 期望字节数
fetch() {
    name="$1"
    want="$2"
    target="$OUT_DIR/$name"

    attempt=1
    while [ "$attempt" -le 40 ]; do
        have=0
        [ -f "$target" ] && have=$(wc -c < "$target" | tr -d ' ')
        if [ "$have" -ge "$want" ]; then
            echo "[fetch] OK $name ($have bytes)"
            return 0
        fi
        echo "[fetch] $name attempt=$attempt have=$have/$want"
        curl -sSL --retry 3 --retry-delay 5 --retry-all-errors --connect-timeout 20 \
             -C - -o "$target" "$BASE/$name" || true
        attempt=$((attempt + 1))
        sleep 2
    done
    echo "[fetch] FAILED $name"
    return 1
}

mkdir -p "$OUT_DIR"
fetch "OvisOCR2-Q4_K_M.gguf" 529297280
fetch "mmproj-F16.gguf" 204987079
echo "[fetch] all done"
ls -la "$OUT_DIR"
