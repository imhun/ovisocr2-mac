"""MPS 实机能力探测：验证 OvisOCR2 推理会走到的关键算子路径。

只读诊断脚本，不修改任何外部状态。
"""

import platform
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F


def section(title: str) -> None:
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def report(label: str, fn) -> bool:
    try:
        fn()
        print(f"  [ OK ] {label}")
        return True
    except Exception as exc:  # noqa: BLE001 - 诊断脚本需要吞掉并报告所有异常
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
        print(f"  [FAIL] {label}: {type(exc).__name__}: {first[:150]}")
        return False


def main() -> None:
    section("0. 环境")
    print("machine        :", platform.machine())
    print("macOS          :", platform.mac_ver()[0])
    print("PyTorch        :", torch.__version__)
    print("mps is_built   :", torch.backends.mps.is_built())
    print("mps is_avail   :", torch.backends.mps.is_available())
    dev = torch.device("mps")
    print("mps device name:", torch.mps.get_device_name() if hasattr(torch.mps, "get_device_name") else "n/a")

    # ---------------------------------------------------------------- 1
    section("1. 基础算子 (fp32)")
    x = torch.randn(64, 128, device=dev)
    w = torch.randn(128, 256, device=dev)
    report("matmul", lambda: x @ w)
    report("add/mul", lambda: (x + 1.0) * 2.0)
    report("softmax(dim=-1)", lambda: torch.softmax(x, dim=-1))
    report("layer_norm", lambda: F.layer_norm(x, (128,)))
    report("silu", lambda: F.silu(x))
    report("gelu", lambda: F.gelu(x))
    report("scaled_dot_product_attention", lambda: F.scaled_dot_product_attention(
        torch.randn(2, 8, 32, 64, device=dev),
        torch.randn(2, 8, 32, 64, device=dev),
        torch.randn(2, 8, 32, 64, device=dev),
    ))

    # ---------------------------------------------------------------- 2
    section("2. DeltaNet / Qwen3.5 关键路径 (fp32)")
    # Qwen3NextGatedDeltaNet 的 causal conv1d (kernel_size=4)
    hidden = torch.randn(1, 512, 256, device=dev)
    conv_w = torch.randn(256, 1, 4, device=dev)
    report("conv1d(groups=C, k=4)", lambda: F.conv1d(hidden, conv_w, groups=256, padding=3))
    report("cumsum(dim=-1)", lambda: torch.cumsum(hidden, dim=-1))
    report("einsum", lambda: torch.einsum("btd,bsd->bts", hidden, hidden))
    report("triu / tril", lambda: (torch.triu(torch.randn(64, 64, device=dev)), torch.tril(torch.randn(64, 64, device=dev))))
    report("repeat_interleave", lambda: torch.randn(4, 8, device=dev).repeat_interleave(2, dim=0))
    report("scatter / index_add", lambda: torch.zeros(64, 8, device=dev).index_add_(
        0, torch.arange(64, device=dev), torch.randn(64, 8, device=dev)))
    report("RMSNorm 手写", lambda: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))

    # ---------------------------------------------------------------- 3
    section("3. dtype 支持 (OvisOCR2 权重是 bf16)")
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        name = str(dtype).replace("torch.", "")
        def _run(dtype=dtype):
            a = torch.randn(32, 64, device=dev, dtype=dtype)
            b = torch.randn(64, 64, device=dev, dtype=dtype)
            return a @ b
        ok = report(f"matmul {name}", _run)
        if ok:
            print(f"         -> dtype 实际保持 : {_run().dtype}")

    # ---------------------------------------------------------------- 4
    section("4. 正确性对照 (MPS vs CPU, fp32)")
    torch.manual_seed(0)
    a_cpu = torch.randn(128, 128)
    b_cpu = torch.randn(128, 128)
    ref = (a_cpu @ b_cpu)
    try:
        got = (a_cpu.to(dev) @ b_cpu.to(dev)).cpu()
        diff = (got - ref).abs().max().item()
        print(f"  matmul 最大绝对误差 : {diff:.3e}  {'OK' if diff < 1e-4 else '⚠ 偏差偏大'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] matmul 对照: {type(exc).__name__}: {exc}")

    # softmax 是 MPS 历史上出问题的重灾区
    s_cpu = torch.randn(4, 16, 512)
    ref_s = torch.softmax(s_cpu, dim=-1)
    try:
        got_s = torch.softmax(s_cpu.to(dev), dim=-1).cpu()
        d = (got_s - ref_s).abs().max().item()
        print(f"  softmax 最大绝对误差: {d:.3e}  {'OK' if d < 1e-4 else '⚠ 偏差偏大'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] softmax 对照: {type(exc).__name__}: {exc}")

    # ---------------------------------------------------------------- 5
    section("5. 性能对照: 1024x1024 矩阵乘 x100")
    for label, device in (("CPU", torch.device("cpu")), ("MPS", dev)):
        try:
            m1 = torch.randn(1024, 1024, device=device)
            m2 = torch.randn(1024, 1024, device=device)
            if device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                m1 = m1 @ m2
                m1 = m1 / m1.norm()  # 防止数值爆炸
            if device.type == "mps":
                torch.mps.synchronize()
            dt = time.perf_counter() - t0
            print(f"  {label}: {dt:.3f}s  ({100 / dt:.1f} iter/s)")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {label} benchmark: {type(exc).__name__}: {str(exc)[:120]}")

    # ---------------------------------------------------------------- 6
    section("6. 迷你 transformer block 前向 (2 次，热身+计时)")
    class Block(nn.Module):
        def __init__(self, d=256, h=8):
            super().__init__()
            self.norm1 = nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3 * d)
            self.proj = nn.Linear(d, d)
            self.norm2 = nn.LayerNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.SiLU(), nn.Linear(4 * d, d))
            self.h = h

        def forward(self, x):
            b, t, d = x.shape
            y = self.norm1(x)
            q, k, v = self.qkv(y).chunk(3, dim=-1)
            q, k, v = (z.view(b, t, self.h, d // self.h).transpose(1, 2) for z in (q, k, v))
            attn = F.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).reshape(b, t, d)
            x = x + self.proj(attn)
            return x + self.mlp(self.norm2(x))

    for label, device in (("CPU", torch.device("cpu")), ("MPS", dev)):
        try:
            blk = Block().to(device).eval()
            inp = torch.randn(1, 512, 256, device=device)
            with torch.no_grad():
                _ = blk(inp)
                if device.type == "mps":
                    torch.mps.synchronize()
                t0 = time.perf_counter()
                for _ in range(10):
                    _ = blk(inp)
                if device.type == "mps":
                    torch.mps.synchronize()
                dt = (time.perf_counter() - t0) / 10
            print(f"  {label}: {dt * 1000:.1f} ms / forward (seq=512, d=256)")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {label} block: {type(exc).__name__}: {str(exc)[:150]}")
            traceback.print_exc(limit=1)

    section("探测结束")


if __name__ == "__main__":
    main()
