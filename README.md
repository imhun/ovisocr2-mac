# OvisOCR2 本地部署（macOS / Intel Mac 实测）

OvisOCR2 是阿里 ATH-MaaS 发布的 0.8B 端到端文档解析模型（基于 Qwen3.5-0.8B），
输入整页文档图片，直接输出自然阅读顺序的 Markdown（含 LaTeX 公式、HTML 表格、
视觉区域坐标）。

本机：`MacBookPro16,1`，Intel Core i9-9880H，AMD Radeon Pro 5500M (8GB) + Intel UHD 630，
macOS 26.7.1，**非 Apple Silicon**。以下结论均来自本机实测。

仓库提供两个脚本：`src/ovisocr2_llama.py` 做单页或批量图片的 OCR，
`src/pdf2md.py` 把整份 PDF / Office 文档转成 Markdown。

---

## 一、结论速览

在这台机器上跑 OvisOCR2，几条路的实际情况：

| 方式 | 状态 | 每页耗时 | 说明 |
| --- | --- | --- | --- |
| **llama.cpp + Metal 补丁** | ✅ **推荐** | **~9-15s** | AMD GPU 全速；构建见 [llamacpp-metal-amd](https://github.com/imhun/llamacpp-metal-amd) |
| llama.cpp 退回 CPU | ✅ 可用 | ~68s | 没有 GPU 构建时的兜底，缩图后能快一截 |
| PyTorch + MPS | ❌ 本机不可行 | — | 版本链死锁，见第三节；Apple Silicon 上可行 |
| 云端 GPU | ✅ | 实时 | 0.8B 模型，任意消费级显卡都够 |

本机的 GPU 加速走的是 llama.cpp，不是 PyTorch——原因在第三节。

---

## 二、快速开始

克隆后一条命令装完：

```bash
git clone --recurse-submodules https://github.com/imhun/ovisocr2-mac.git
cd ovisocr2-mac
./scripts/build.sh

# 国内网络下载权重慢，指定镜像
HF_BASE=https://hf-mirror.com/Abiray/OvisOCR2-GGUF/resolve/main ./scripts/build.sh

# 权重已下好 / 只更新代码
./scripts/build.sh --skip-models
```

脚本依次完成：环境检查 → Python 依赖 → 下载 GGUF 权重 → 准备 llama.cpp → 冒烟测试。
每一步都会先检查是否已经完成，可以反复执行；首次编译 5-15 分钟，之后重跑直接跳过。
全部参数见 `./scripts/build.sh --help`。

llama.cpp 那一段不是本仓库自己做的，而是调用
[llamacpp-metal-amd](https://github.com/imhun/llamacpp-metal-amd)——补丁、pin 的版本和
编译开关都在那边，产物落在本仓库的 `tmp/llama.cpp/`。它作为 submodule 挂在
`third_party/` 下，克隆时忘了 `--recurse-submodules` 也没关系，脚本会自己补拉。
如果那个仓库已经把二进制装到全局，这里会直接复用，不再编译。

冒烟测试会生成一张小测试图跑一次真实 OCR，检查三件事：输出非空、没有 `@@@@` 之类的
退化内容、日志里能读到 Metal 设备。

---

## 三、PyTorch + MPS 为什么在这台机器上不行

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

这点值得记录 —— **MPS 后端能跑，而且比 CPU 快**，只是被上面的版本链拦在门外：

| 测试项 | CPU | MPS | 结果 |
| --- | --- | --- | --- |
| 1024×1024 矩阵乘 ×100 | 0.561 s | 0.435 s | MPS 快 1.29× |
| 迷你 Transformer block (seq=512, d=256) | 4.4 ms | 2.9 ms | MPS 快 1.52× |
| matmul 数值误差 | — | 2.3e-05 | 正常 |
| softmax 数值误差 | — | 7.5e-09 | 正常 |

DeltaNet 关键算子（depthwise `conv1d`、`cumsum`、`einsum`、`triu`、`index_add`、
手写 RMSNorm）在 MPS 上全部正确，相对误差 ~1e-7。唯一硬件限制是 **MPS 不支持
bfloat16**（`TypeError: BFloat16 is not supported on MPS`），而 OvisOCR2 的权重正是
bf16，所以不管在哪台机器上都得转成 fp16 加载。

复现探测：

```bash
.venv/bin/python scripts/mps_probe.py
```

---

## 四、可用的几条路

### 方案 A：本机 llama.cpp（AMD 独显，推荐）

```bash
./scripts/build.sh                                     # 一键构建

.venv/bin/python src/ovisocr2_llama.py page.jpg -o out                  # 默认长边 1600
.venv/bin/python src/ovisocr2_llama.py scans/ -o out --max-side 2000    # 更清晰
```

脚本按 `tmp/llama.cpp/build-metal/bin/llama-mtmd-cli` → 全局安装 → `PATH` → 官方构建
的顺序挑二进制，再用 `-ngl 99` 把模型全部放到 GPU。

`--max-side` 默认 1600、上限压到 2000，是因为 macOS 的 GPU watchdog 在长边超过 2400
之后会让输出损坏。这条限制的来龙去脉、以及各后端的实测对比，都在
[llamacpp-metal-amd](https://github.com/imhun/llamacpp-metal-amd)。

实测输出（本页测试图）：

```markdown
OvisOCR2 Deployment Test

This page verifies end-to-end document parsing on local hardware. ...

$$ \mathrm{E}=\mathrm{m}\ \mathrm{c}^{2} $$

<table><thead><tr><td>Model</td><td>Params</td><td>OmniDocBench</td></tr>...

- reading order preserved
```

### 方案 B：Apple Silicon Mac（PyTorch + MPS）

M1 及以上即可，建议 M2+（原生 bf16 支持）：

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements-torch.txt   # torch / transformers / pillow
.venv/bin/python src/ovisocr2_pytorch.py page.jpg -o out
```

Intel Mac 上这份依赖装不上，属于预期行为，原因见第三节。

### 方案 C：云端 / 远程 GPU

模型仅 0.8B，官方推荐 vLLM：

```bash
pip install "vllm==0.22.1" pillow
```

按 HF 模型卡的 `OvisOCR2Parser` 示例调用（vLLM 不支持 macOS）。

---

## 五、整份文档转 Markdown

单页图片是一回事，把一篇论文转干净是另一回事：正文交给 OCR 太慢，交给文本提取又丢
表格和图。所以 `src/pdf2md.py` 让两个工具各干自己擅长的部分。

**PDF**：先用 anydoc 出正文（毫秒级，文本层和字体元数据最完整），再用 pymupdf 找出
含图表的页，只对这些页跑 OvisOCR2，最后按文本锚点把补充内容插回正文对应位置。

```bash
.venv/bin/python src/pdf2md.py paper.pdf -o out/
.venv/bin/python src/pdf2md.py paper.pdf -o out/ --pages 6,12   # 手动指定页
.venv/bin/python src/pdf2md.py scan.pdf  -o out/ --no-anydoc    # 扫描件，整页 OCR
```

挑页用三个信号，命中一个就算：位图对象、pymupdf 识别的表格、**矢量绘图数量 ≥ 15**。
第三个是关键——论文里的状态机流程图常常是矢量画的，页面上一个位图都没有，只测位图
会整页漏掉。

24 页的论文实测：命中 4 页，总共 1 分 39 秒，产出 9.2 万字符的 Markdown 和 3 张裁好的
图。全量 OCR 这 24 页要 8 分钟以上。锚点匹配在 0.95 到 1.00 之间。

**Office 格式**（docx / pptx / xlsx）走另一条路：anydoc 直接给结构化对象，表格的合并
单元格、图片的原始字节都在，不需要渲染也不需要 OCR。同一个测试 docx 是 14 毫秒。

两个 API 上的坑记一下：anydoc 的 `to_document()` 不接受 PDF（会报
`UnsupportedError`）；`to_markdown()` 处理 docx 时标题、列表、表格都对，但**图片会完全
消失，连占位符都不留**。

---

## 六、目录说明

| 路径 | 说明 |
| --- | --- |
| `src/pdf2md.py` | **通用文档→MD**：PDF 走混合流程，Office 走结构化通道 |
| `src/ovisocr2_llama.py` | 单页/批量 OCR：llama.cpp 推理、自动缩图、think 清理、损坏检测 |
| `src/ovisocr2_pytorch.py` | PyTorch / MPS 推理（需 Apple Silicon） |
| `third_party/llamacpp-metal-amd` | **submodule**：llama.cpp 的 Metal 补丁构建，独立仓库维护 |
| `scripts/build.sh` | 一键构建：环境检查、依赖、权重、调 submodule 构建 llama.cpp、冒烟测试 |
| `scripts/mps_probe.py` | MPS 能力与性能探测，可复现第三节的数据 |
| `scripts/fetch_gguf.sh` | GGUF 下载脚本（断点续传 + 重试，`HF_BASE` 可指向 hf-mirror） |
| `requirements.txt` | llama.cpp 路径的运行依赖（pillow / pymupdf / firecrawl-anydoc） |
| `requirements-torch.txt` | PyTorch 路径的依赖（torch / transformers，需 Apple Silicon） |
| `models/` | GGUF 权重（`OvisOCR2-Q4_K_M.gguf` + `mmproj-F16.gguf`，不入库） |
| `tmp/` | 工作目录（不入库）：llama.cpp 源码与构建、测试产物 |

常用参数：

```bash
.venv/bin/python src/ovisocr2_llama.py page.jpg -o out -v   # 透传 llama.cpp 日志
.venv/bin/python src/ovisocr2_llama.py --help
```

`-v` 在排查输出异常时有用：它会把 llama.cpp 的原始日志打出来，能直接看到接管的是不是
Metal 后端，以及视觉编码花了多久。

---

## 七、参考

- HF 模型卡：https://huggingface.co/ATH-MaaS/OvisOCR2
- 技术报告：https://arxiv.org/abs/2607.13639
- transformers Qwen3.5 文档：https://huggingface.co/docs/transformers/model_doc/qwen3_5
- GGUF 量化：https://huggingface.co/Abiray/OvisOCR2-GGUF
- llama.cpp 的 Metal 补丁构建：https://github.com/imhun/llamacpp-metal-amd
