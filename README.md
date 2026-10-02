# OvisOCR2 本地部署（macOS / Intel Mac 实测）

OvisOCR2 是阿里 ATH-MaaS 发布的 0.8B 端到端文档解析模型（基于 Qwen3.5-0.8B），
输入整页文档图片，直接输出自然阅读顺序的 Markdown（含 LaTeX 公式、HTML 表格、视觉区域坐标）。

本机：`MacBookPro16,1`，Intel Core i9-9880H，AMD Radeon Pro 5500M (8GB) + Intel UHD 630，
macOS 26.7.1，**非 Apple Silicon**。以下结论均来自本机实测。

---

## 一、结论速览

| 方案 | 状态 | 每页耗时 | 说明 |
| --- | --- | --- | --- |
| PyTorch + MPS | ❌ 不可行 | — | 版本死锁，见第二节 |
| **llama.cpp + ToshLLM Metal 补丁** | ✅ **推荐** | **~9-15s** | **AMD GPU 全速，视觉编码仅 6.2s** |
| llama.cpp CPU (Accelerate) | ✅ 可用 | ~68s | 回退方案；缩图后视觉编码 159s → 26s |
| llama.cpp Metal (stock) | ❌ 输出损坏 | 39s 后崩 | GPU Timeout + `@@@@` 乱码 |
| llama.cpp Vulkan (MoltenVK) | ❌ 输出损坏 | ~27s | 输出全为 `1`，CLIP `SOFT_MAX` 不支持 |

**当前可立即使用**：`src/ovisocr2_llama.py`，默认走 ToshLLM Metal 补丁版（AMD GPU 加速）。

---

## 二、PyTorch + MPS 为何不可行

不是配置问题，是版本链死锁：

```
OvisOCR2 (Qwen3.5 架构)
   └─ 需要 transformers >= 5.0        （唯一支持 qwen3_5 的版本线）
        └─ 硬性要求 torch >= 2.5      （import_utils 写死，且大量使用 2.4+/2.5+ API）
             └─ 但本机 torch 上限 = 2.2.2
                  └─ PyTorch 自 2.3.0 起不再发布 macOS x86_64 wheel
```

### 证据

```console
$ uv pip install --dry-run "torch>=2.5"
error: No solution found when resolving dependencies
  cause: Because torch>=2.5.0 has no wheels with a matching platform tag
         (e.g., `macosx_26_0_x86_64`) ...
  hint: Wheels are available for `torch` (v2.14.0) on the following platforms:
        manylinux_2_28_aarch64, manylinux_2_28_x86_64, macosx_14_0_arm64, win_amd64
```

`torch==2.3.0` / `2.4.0` 同样只有 `macosx_11_0_arm64`。本机能装的最高版本就是 **2.2.2**。

transformers 5.17.0 的版本门禁与 API 依赖规模：

```console
$ python -c "import transformers"
[transformers] Disabling PyTorch because PyTorch >= 2.5 is required but found 2.2.2
```

强行放宽检查后立刻撞墙：`ImportError: cannot import name 'DTensor' from 'torch.distributed.tensor'`

| API（引入版本） | transformers 5.17 中出现次数 |
| --- | --- |
| `DTensor`（2.4+） | 85 |
| `device_mesh`（2.4+） | 73 |
| `flex_attention`（2.5+） | 53 |
| `torch.compile` | 156 |
| `torch.float8`（2.4+） | 13 |

几百处依赖，逐个打补丁不可行。

### 附带发现：MPS 在这台机器上其实是可用的

这点值得记录 —— **MPS 后端能跑，而且比 CPU 快**（只是版本链拦住了）：

| 测试项 | CPU | MPS | 结果 |
| --- | --- | --- | --- |
| 1024×1024 矩阵乘 ×100 | 0.561 s | 0.435 s | MPS 快 1.29× |
| 迷你 Transformer block (seq=512, d=256) | 4.4 ms | 2.9 ms | MPS 快 1.52× |
| matmul 数值误差 | — | 2.3e-05 | 正常 |
| softmax 数值误差 | — | 7.5e-09 | 正常 |

DeltaNet 关键算子（depthwise `conv1d`、`cumsum`、`einsum`、`triu`、`index_add`、手写 RMSNorm）
在 MPS 上全部正确，相对误差 ~1e-7。唯一硬件限制：**MPS 不支持 bfloat16**
（`TypeError: BFloat16 is not supported on MPS`），而 OvisOCR2 权重正是 bf16，
因此在 Apple Silicon 或本机上都必须以 fp16 加载。

---

## 三、llama.cpp + AMD GPU：实测全部失败

前置条件：Homebrew 版 llama.cpp 在 Intel Mac 上**没有编译 Metal**
（`--list-devices` 只有 `BLAS: Accelerate`），因为 **Homebrew 自 2026 年 9 月起
已放弃 Intel x86_64 macOS 支持**（不再提供 bottle）。所以下面两个后端都是源码编译的。

### 3.1 Metal（stock，`-DGGML_METAL=ON`）

设备能被正确识别：

```
Available devices:
  MTL0: AMD Radeon Pro 5500M (8176 MiB, 8175 MiB free)
  BLAS: Accelerate (0 MiB, 0 MiB free)
```

视觉编码确实快了很多（159s → 39.4s），但**输出完全损坏**：

```
E ggml_metal_synchronize: error: command buffer 0 failed with status 5
E error: Caused GPU Timeout Error (00000002:kIOAccelCommandBufferCallbackErrorTimeout)
...
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@...
```

根因（对应 llama.cpp issue #15228）：llama.cpp 用
`newBufferWithBytesNoCopy(... options:MTLResourceStorageModeShared ...)`
（`ggml/src/ggml-metal/ggml-metal-device.m:2214`）把权重映射到共享内存。
在**独显**上，这意味着 GPU 每次推理都要通过 PCIe 从系统内存读权重，
造成带宽瓶颈与 GPU 超时。该 issue 已被官方关闭为 **not planned**。

### 3.2 Vulkan（MoltenVK）

依赖：`molten-vk` + `vulkan-loader` + `shaderc`(glslc) + `spirv-headers` + `glslang`。

设备也能识别：

```
Vulkan0: AMD Radeon Pro 5500M (8176 MiB)
Vulkan1: Intel(R) UHD Graphics 630 (65536 MiB)
```

但输出全是 `1`，且日志明确报告视觉编码器缺算子：

```
W resolve_fused_ops: layer 3 is assigned to device Vulkan0 but Flash Attention is assigned to device CPU
W warmup: WARNING: flash attention not supported by Vulkan0
W warmup: WARNING: the CLIP graph uses unsupported operators by the backend
W warmup: list of unsupported ops (backend=Vulkan0):
W warmup:   SOFT_MAX: type = f32, ne = [8580 8580 12 1]
```

这与 issue #20104（Vulkan on Intel Macs produce gibberish）一致，该 issue 被标记为
#20029 的重复项。Zsh 侧无需额外设置，但运行前需要指定 ICD：

```bash
export VK_ICD_FILENAMES=/usr/local/etc/vulkan/icd.d/MoltenVK_icd.json
```

### 3.3 ToshLLM 补丁系列 —— ✅ 已适配成功

这是本仓库最终采用的方案。做法是把 ToshLLM 验证过的补丁系列应用到它 pin 的
llama.cpp commit 上，重建整个 Metal 后端。补丁已随仓库放在 `patches/llama/`
（ToshLLM v0.87.13，119 个文件，逐字节未修改，来源见 `patches/README.md`）：

```bash
# 1. 取它 pin 的 commit（补丁是针对这个版本验证的）
git clone --filter=blob:none https://github.com/ggml-org/llama.cpp tmp/tosh-llama
cd tmp/tosh-llama && git checkout -qf 9575389609d6f8437de0b205561a4824d217c409

# 2. 按文件名数字顺序应用全部 119 个补丁（0 失败）
#    ../../patches/llama 是相对 tmp/tosh-llama 的路径
for p in ../../patches/llama/*.patch; do
  git apply "$p"
done

# 3. 构建（Metal + Accelerate）
cmake -B build-tosh -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DTOSH_ENABLE_DYNAMIC_MOE=ON \
  -DGGML_NATIVE=OFF -DCMAKE_OSX_ARCHITECTURES=x86_64 -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
  -DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON \
  -DGGML_AVX_VNNI=OFF -DGGML_AVX512=OFF -DLLAMA_OPENSSL=OFF
cmake --build build-tosh -j 8 -t llama-mtmd-cli
```

启动时会看到补丁生效的标志 —— stock 版没有的 SIMD 宽度探测与独显识别：

```
ggml_metal: device 0: AMD Radeon Pro 5500M (peer group 0, not bridged)
            probed SIMD-group width = 32 (32 = Apple/AMD RDNA, 64 = AMD GCN/Vega)
```

**同一条命令的前后对比**（`-ngl 99`，同一张 1240×1754 页面）：

| 后端 | 视觉编码 | 总耗时 | 输出 |
| --- | --- | --- | --- |
| Metal (stock) | 39.4 s | 崩溃 | `@@@@@@` + GPU Timeout |
| Vulkan (stock) | ~27 s | 41 s | 全为 `1` |
| CPU | 159 s | — | 正确但慢 |
| **ToshLLM Metal** | **6.2 s** | **15 s** | **完全正确** |

稳定性：连续 3 次均为 12 s、0 乱码、编码 5.63 s。

**唯一约束是分辨率上限**（macOS 的 GPU watchdog，无法从代码侧绕过）：

| 长边 | 视觉编码 | 结果 |
| --- | --- | --- |
| 1600 | 3.1 s | ✅ |
| 2000 | 7.6 s | ✅ |
| 2400 | 18.2 s | ⚠️ 出现 GPU timeout，输出侥幸正确 |
| 2800+ | 18.4 s | ❌ 输出损坏 |
| 3508 | 18.6 s | ❌ 输出损坏 |

因此 `src/ovisocr2_llama.py` 默认 `--max-side 1600`，并把超过 2000 的值强制压到 2000，
同时对输出做损坏检测（识别 `@@@@` 与重复的 `## 1`）并告警。

> 备注：`patches/llama/` 下共 119 个补丁，光 `0001` 和 `0002` 两个就有 15788 行，
> 是针对 upstream `9575389609d6` 的完整 Metal 后端重建；换 commit 需要重新适配。
> 另有 issue #15228 的早期 gist（`ggml-metal-optimized-4.m`），针对 2025-08 的
> `ggml-metal.m` 结构，在当前代码上已无法直接套用。

---

## 四、可用方案

### 方案 A：本机 llama.cpp + ToshLLM Metal 补丁（AMD GPU，推荐）

```bash
python src/ovisocr2_llama.py page.jpg -o out           # 默认 1600，约 9s/页
python src/ovisocr2_llama.py scans/ -o out --max-side 2000   # 更清晰，约 15s/页
```

脚本会按 `tmp/tosh-llama/build-tosh/bin/llama-mtmd-cli` → `PATH` → 官方构建
的顺序自动挑选二进制，并用 `-ngl 99` 把模型全部放到 GPU。

回退到 CPU 时（没有 GPU 构建）用缩图，视觉编码耗时与分辨率成正比：

| 预处理长边 | 视觉编码耗时 | 相对原图 |
| --- | --- | --- |
| 1754（原图） | 159 s | 1× |
| 1000 | 26 s | 6× 加速 |

实测输出（本页测试图）：

```markdown
OvisOCR2 Deployment Test

This page verifies end-to-end document parsing on local hardware. ...

$$ \mathrm{E}=\mathrm{m}\ \mathrm{c}^{2} $$

<table><thead><tr><td>Model</td><td>Params</td><td>OmniDocBench</td></tr>...

- reading order preserved
```

### 方案 B：Apple Silicon Mac（唯一能满足"PyTorch + MPS"的路径）

M1 及以上即可，建议 M2+（原生 bf16 支持）。环境：

```bash
uv venv --python 3.12 .venv
uv pip install "torch>=2.5" transformers pillow
python src/ovisocr2_pytorch.py page.jpg -o out   # 见第五节
```

### 方案 C：云端 / 远程 GPU

模型仅 0.8B，任意消费级 GPU 即可实时推理，官方推荐 vLLM：

```bash
pip install "vllm==0.22.1" pillow
```

按 HF 模型卡的 `OvisOCR2Parser` 示例调用（vLLM 不支持 macOS）。

---

## 五、目录说明

| 路径 | 说明 |
| --- | --- |
| `src/pdf2md.py` | **通用文档→MD**：PDF 走 anydoc 正文 + OvisOCR2 补图表页；Office 走 anydoc 结构化通道（表格/图片无损，无需 OCR） |
| `src/ovisocr2_llama.py` | **可用**：llama.cpp CPU 推理（含自动缩图、think 清理） |
| `src/ovisocr2_pytorch.py` | PyTorch / MPS 推理（需 Apple Silicon，本机跑不了） |
| `patches/llama/` | ToshLLM v0.87.13 的 119 个 llama.cpp 补丁（GPL-3.0-or-later，来源见 `patches/README.md`） |
| `scripts/mps_probe.py` | MPS 能力与性能探测，可复现第二节 2 的数据 |
| `scripts/fetch_gguf.sh` | GGUF 下载脚本（断点续传 + 重试，`HF_BASE` 可指向 hf-mirror 镜像） |
| `models/` | GGUF 权重（`OvisOCR2-Q4_K_M.gguf` + `mmproj-F16.gguf`） |
| `tmp/` | 工作目录（不入库）：官方 llama.cpp 源码与构建、ToshLLM 补丁版构建、测试产物 |

复现探测：

```bash
.venv/bin/python scripts/mps_probe.py
```

---

## 六、参考

- HF 模型卡：https://huggingface.co/ATH-MaaS/OvisOCR2
- 技术报告：https://arxiv.org/abs/2607.13639
- transformers Qwen3.5 文档：https://huggingface.co/docs/transformers/model_doc/qwen3_5
- GGUF 量化：https://huggingface.co/Abiray/OvisOCR2-GGUF
- Metal/AMD 问题：https://github.com/ggml-org/llama.cpp/issues/15228
- Vulkan 乱码：https://github.com/ggml-org/llama.cpp/issues/20104
