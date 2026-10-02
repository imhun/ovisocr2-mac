# ToshLLM llama.cpp 补丁系列

本目录是 [ToshLLM](https://github.com/engeldlgado/toshllm) 补丁系列的副本，用来复现
AMD 独显上的 Metal 加速构建。

| 项 | 值 |
| --- | --- |
| 来源 | https://github.com/engeldlgado/toshllm |
| 版本 | v0.87.13 |
| commit | `8af5d379d720560c1f0a6d7e38468b5b78a673db` |
| 取用日期 | 2026-10-02 |
| 内容 | 原仓库 `patches/llama/`，119 个文件，逐字节未修改 |
| 目标 upstream | llama.cpp `9575389609d6f8437de0b205561a4824d217c409` |

## 许可

见同目录 `COPYRIGHT`（原仓库文件，未修改）：

```
Every patch in this directory is
Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
SPDX-License-Identifier: GPL-3.0-or-later
```

补丁打在这些文件上，文件本身仍归各自作者、沿用 MIT 许可；补丁引入的改动是 ToshLLM
的作品。这里只做了原样转载，没有修改任何补丁内容。

## 应用方式

补丁绑定 upstream 的那个 commit，换版本需要重新适配。按文件名数字顺序应用：

```bash
git clone --filter=blob:none https://github.com/ggml-org/llama.cpp tmp/tosh-llama
cd tmp/tosh-llama && git checkout -qf 9575389609d6f8437de0b205561a4824d217c409

for p in "$OLDPWD"/patches/llama/*.patch; do
  git apply "$p"
done
```

119 个补丁实测全部应用成功，0 失败。构建命令见仓库根目录 `README.md` 第 3.3 节。

## 规模

`0001` 和 `0002` 两个文件加起来就有 15788 行，整个系列相当于把 Metal 后端重写了一遍。
它们是基线补丁（按文件划分：`0001` 是 Metal kernel，`0002` 是后端宿主代码），编号更大的
才是后续的独立改动。ToshLLM 上游还有一个 `patches/README.md` 记录了补丁的维护规则，
需要改补丁时值得先读一遍。
