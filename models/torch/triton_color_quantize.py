"""Fused Triton kernel for color quantization.

Replaces the ``torch.compile``-d ``TorchColorQuantization`` with a
hand-written Triton kernel that performs the entire forward pass —
squared distances, softmax (K=4), and weighted sum (C=3) — in a
**single kernel launch** per call.  This eliminates 2–3 intermediate
kernel launches that Inductor generates when decomposing ``einsum`` and
``softmax`` for such tiny fixed dimensions.

The backward pass is also fused into a single kernel via
``torch.autograd.Function``.

Palette
-------
K=4 colors (Black, Red, Green, Blue) with C=3 channels, each in [-1, 1]:

    Black = (-1, -1, -1)
    Red   = ( 1, -1, -1)
    Green = (-1,  1, -1)
    Blue  = (-1, -1,  1)

Because K and C are compile-time constants the inner loops fully unroll
and all intermediates stay in registers.
"""

from __future__ import annotations

from typing import cast

import triton  # type: ignore[import-untyped]
import triton.language as tl  # type: ignore[import-untyped]
import torch

# ── Fixed palette constants ──────────────────────────────────────────
# Flattened (K*C,) = 12 values.  Indexed as colors[k*3 + c].
_PALETTE_FLAT: list[float] = [
    -1.0, -1.0, -1.0,   # Black
     1.0, -1.0, -1.0,   # Red
    -1.0,  1.0, -1.0,   # Green
    -1.0, -1.0,  1.0,   # Blue
]
# ||c||² for each of the 4 colors (all equal to 3.0 for this palette).
_C_SQ: list[float] = [3.0, 3.0, 3.0, 3.0]

K: int = 4  # palette size
C: int = 3  # channels


# ── Forward kernel ───────────────────────────────────────────────────
@triton.jit  # type: ignore[misc]
def _color_quant_fwd_kernel(
    X_ptr: torch.Tensor,        # (B, 3, H, W) input — contiguous or channels_last
    OUT_ptr: torch.Tensor,      # (B, 3, H, W) output
    WEIGHTS_ptr: torch.Tensor,  # (B, 4, H, W) saved softmax weights (for backward)
    stride_b: int,     # stride along batch dimension
    stride_c: int,     # stride along channel dimension
    stride_h: int,     # stride along height dimension
    stride_w: int,     # stride along width dimension
    HW: tl.constexpr,        # H * W
    inv_temp: tl.constexpr,  # 1 / temperature
    BLOCK: tl.constexpr,     # tile size over the HW dimension
    TRAINING: tl.constexpr,  # 1 for soft quantization, 0 for hard
):
    """One program processes one (batch, hw_tile) rectangle."""
    pid = tl.program_id(0)  # type: ignore[misc]
    num_hw_blocks = tl.cdiv(HW, BLOCK)  # type: ignore[misc]
    b = pid // num_hw_blocks
    hw_block = pid % num_hw_blocks
    hw_offs = hw_block * BLOCK + tl.arange(0, BLOCK)
    mask = hw_offs < HW

    # Compute h, w from flat hw index for strided access.
    W_dim = stride_h // stride_w  # infer W from strides when contiguous
    # Actually we receive strides directly; compute element offsets.
    h_idx = hw_offs // (stride_h // stride_w) if stride_w == 1 else hw_offs
    w_idx = hw_offs % (stride_h // stride_w) if stride_w == 1 else hw_offs

    # For both contiguous (NCHW) and channels_last (NHWC) we can compute
    # the base pointer for batch b and then index using strides.
    # But for channels_last stride_c=1 and stride_w=C, stride_h=W*C.
    # We'll compute offsets directly from the flat hw index.

    # We need the spatial position (h, w) to compute the pointer offsets.
    # Given the strides, the offset for element (b, c, h, w) is:
    #   b * stride_b + c * stride_c + h * stride_h + w * stride_w
    # We can use hw_offs to derive h and w if we know W.
    # For NCHW: stride_c = H*W, stride_h = W, stride_w = 1 → W = stride_h
    # For NHWC: stride_c = 1, stride_h = W*C, stride_w = C → W = stride_h // C

    # Determine spatial W from strides (works for both NCHW and NHWC).
    # NCHW: stride_w=1, W = stride_h / stride_w = stride_h
    # NHWC: stride_w=C, W = stride_h / stride_w
    W_spatial = stride_h // stride_w
    h_vals = hw_offs // W_spatial
    w_vals = hw_offs % W_spatial

    base = b * stride_b

    # ── Load 3 input channels ──
    x0 = tl.load(X_ptr + base + 0 * stride_c + h_vals * stride_h + w_vals * stride_w, mask=mask, other=0.0)
    x1 = tl.load(X_ptr + base + 1 * stride_c + h_vals * stride_h + w_vals * stride_w, mask=mask, other=0.0)
    x2 = tl.load(X_ptr + base + 2 * stride_c + h_vals * stride_h + w_vals * stride_w, mask=mask, other=0.0)

    # ── ||x||² ──
    x_sq = x0 * x0 + x1 * x1 + x2 * x2

    # ── Squared distances to each of the 4 palette colors ──
    # color 0: Black (-1,-1,-1), ||c||²=3, dot = -x0 -x1 -x2
    dot0 = -x0 - x1 - x2
    d0 = x_sq + 3.0 - 2.0 * dot0

    # color 1: Red (1,-1,-1), ||c||²=3, dot = x0 -x1 -x2
    dot1 = x0 - x1 - x2
    d1 = x_sq + 3.0 - 2.0 * dot1

    # color 2: Green (-1,1,-1), ||c||²=3, dot = -x0 +x1 -x2
    dot2 = -x0 + x1 - x2
    d2 = x_sq + 3.0 - 2.0 * dot2

    # color 3: Blue (-1,-1,1), ||c||²=3, dot = -x0 -x1 +x2
    dot3 = -x0 - x1 + x2
    d3 = x_sq + 3.0 - 2.0 * dot3

    if TRAINING:
        # ── Softmax over 4 distances ──
        neg_d0 = -d0 * inv_temp
        neg_d1 = -d1 * inv_temp
        neg_d2 = -d2 * inv_temp
        neg_d3 = -d3 * inv_temp

        m = tl.maximum(tl.maximum(neg_d0, neg_d1), tl.maximum(neg_d2, neg_d3))
        e0 = tl.exp(neg_d0 - m)
        e1 = tl.exp(neg_d1 - m)
        e2 = tl.exp(neg_d2 - m)
        e3 = tl.exp(neg_d3 - m)
        s = e0 + e1 + e2 + e3
        w0 = e0 / s
        w1 = e1 / s
        w2 = e2 / s
        w3 = e3 / s

        # ── Weighted sum → output channels ──
        # out_c = sum_k w_k * color[k, c]
        # Black=(-1,-1,-1), Red=(1,-1,-1), Green=(-1,1,-1), Blue=(-1,-1,1)
        out0 = -w0 + w1 - w2 - w3   # channel 0
        out1 = -w0 - w1 + w2 - w3   # channel 1
        out2 = -w0 - w1 - w2 + w3   # channel 2

        # ── Save weights for backward ──
        # weights are stored as (B, K, H, W) contiguous
        w_base = b * 4 * HW + hw_offs
        tl.store(WEIGHTS_ptr + w_base,            w0, mask=mask)
        tl.store(WEIGHTS_ptr + w_base + HW,       w1, mask=mask)
        tl.store(WEIGHTS_ptr + w_base + 2 * HW,   w2, mask=mask)
        tl.store(WEIGHTS_ptr + w_base + 3 * HW,   w3, mask=mask)
    else:
        # ── Hard quantization: argmin ──
        # Start with color 0
        min_d = d0
        best = tl.full(hw_offs.shape, 0, dtype=tl.int32)
        cond1 = d1 < min_d
        min_d = tl.where(cond1, d1, min_d)
        best = tl.where(cond1, 1, best)
        cond2 = d2 < min_d
        min_d = tl.where(cond2, d2, min_d)
        best = tl.where(cond2, 2, best)
        cond3 = d3 < min_d
        best = tl.where(cond3, 3, best)

        # Map index to color: 0→(-1,-1,-1), 1→(1,-1,-1), 2→(-1,1,-1), 3→(-1,-1,1)
        # channel c = -1 unless best == c+1 (for c=0→Red, c=1→Green, c=2→Blue)
        # Actually: for all colors, base is -1 for all channels.
        # Then add +2 for the "on" channel:
        #   color 1 (Red):   ch0 += 2
        #   color 2 (Green): ch1 += 2
        #   color 3 (Blue):  ch2 += 2
        #   color 0 (Black): no +2
        minus_one = tl.full(hw_offs.shape, -1.0, dtype=tl.float32)
        two = tl.full(hw_offs.shape, 2.0, dtype=tl.float32)
        zero_f = tl.full(hw_offs.shape, 0.0, dtype=tl.float32)
        out0 = minus_one + tl.where(best == 1, two, zero_f)
        out1 = minus_one + tl.where(best == 2, two, zero_f)
        out2 = minus_one + tl.where(best == 3, two, zero_f)

    # ── Store output ──
    tl.store(OUT_ptr + base + 0 * stride_c + h_vals * stride_h + w_vals * stride_w, out0, mask=mask)
    tl.store(OUT_ptr + base + 1 * stride_c + h_vals * stride_h + w_vals * stride_w, out1, mask=mask)
    tl.store(OUT_ptr + base + 2 * stride_c + h_vals * stride_h + w_vals * stride_w, out2, mask=mask)


# ── Backward kernel ─────────────────────────────────────────────────
@triton.jit  # type: ignore[misc]
def _color_quant_bwd_kernel(
    GRAD_OUT_ptr: torch.Tensor,  # (B, 3, H, W) upstream gradient
    WEIGHTS_ptr: torch.Tensor,   # (B, 4, H, W) saved softmax weights
    GRAD_IN_ptr: torch.Tensor,   # (B, 3, H, W) gradient w.r.t. input x
    stride_b: int,
    stride_c: int,
    stride_h: int,
    stride_w: int,
    HW: tl.constexpr,
    inv_temp: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Backward of fused color quantization.

    Forward:  out = W @ colors  where W = softmax(-dist/τ)
    We need dx from d_out.

    Chain rule:
      d_out/d_x = d_out/d_W · d_W/d_dist · d_dist/d_x

    Let g = grad_output (B, 3, H, W).

    d_out/d_W_k = sum_c g_c · color[k,c]     →  (B, K, H, W)
    d_W/d_dist  = softmax jacobian: dW_k/d_neg_d_j = W_k(δ_kj - W_j) / τ
    d_dist_k/d_x_c = 2(x_c - color[k,c])

    Combining:
      grad_x_c = sum_k [sum_j dL/d_neg_d_j] · (-1/τ) · W_k · d_dist_k/d_x_c
               + sum_k W_k · ... (softmax jacobian chain)

    More precisely, let:
      p_k = dL/dW_k = sum_c grad_c · color[k,c]
      q_k = (1/τ) · W_k · (p_k - sum_j W_j · p_j)   (softmax backward)
      grad_x_c = sum_k q_k · d(-dist_k)/d(x_c)
               = sum_k q_k · 2 · color[k,c]           (since d_dist = -2·color dotted)

    Wait, let's derive carefully:
      dist_k = ||x||² + ||c_k||² - 2·x·c_k
      neg_d_k = -dist_k / τ
      W = softmax(neg_d)
      out_c = sum_k W_k · color[k,c]

    dL/d(neg_d_k) = sum_j dL/dW_j · dW_j/d(neg_d_k)
                  = sum_j p_j · W_j · (δ_jk - W_k)
                  = W_k · (p_k - sum_j W_j · p_j)

    dL/d(x_c) = sum_k dL/d(neg_d_k) · d(neg_d_k)/d(x_c)
    neg_d_k = -(||x||² + ||c_k||² - 2·x·c_k) / τ
    d(neg_d_k)/d(x_c) = -(2·x_c - 2·c_k[c]) / τ = 2(c_k[c] - x_c) / τ

    So: dL/d(x_c) = sum_k [ W_k · (p_k - dot_wp) ] · 2(c_k[c] - x_c) / τ
    where dot_wp = sum_j W_j · p_j
    """
    pid = tl.program_id(0)  # type: ignore[misc]
    num_hw_blocks = tl.cdiv(HW, BLOCK)  # type: ignore[misc]
    b = pid // num_hw_blocks
    hw_block = pid % num_hw_blocks
    hw_offs = hw_block * BLOCK + tl.arange(0, BLOCK)
    mask = hw_offs < HW

    W_spatial = stride_h // stride_w
    h_vals = hw_offs // W_spatial
    w_vals = hw_offs % W_spatial
    base = b * stride_b

    # Load grad_output channels
    g0 = tl.load(GRAD_OUT_ptr + base + 0 * stride_c + h_vals * stride_h + w_vals * stride_w, mask=mask, other=0.0)
    g1 = tl.load(GRAD_OUT_ptr + base + 1 * stride_c + h_vals * stride_h + w_vals * stride_w, mask=mask, other=0.0)
    g2 = tl.load(GRAD_OUT_ptr + base + 2 * stride_c + h_vals * stride_h + w_vals * stride_w, mask=mask, other=0.0)

    # Load saved softmax weights
    w_base = b * 4 * HW + hw_offs
    w0 = tl.load(WEIGHTS_ptr + w_base,          mask=mask, other=0.0)
    w1 = tl.load(WEIGHTS_ptr + w_base + HW,     mask=mask, other=0.0)
    w2 = tl.load(WEIGHTS_ptr + w_base + 2 * HW, mask=mask, other=0.0)
    w3 = tl.load(WEIGHTS_ptr + w_base + 3 * HW, mask=mask, other=0.0)

    # dL/dW_k = sum_c grad_c * color[k,c]
    # p0 = g0*(-1) + g1*(-1) + g2*(-1) = -g0 -g1 -g2
    # p1 = g0*(1)  + g1*(-1) + g2*(-1) =  g0 -g1 -g2
    # p2 = g0*(-1) + g1*(1)  + g2*(-1) = -g0 +g1 -g2
    # p3 = g0*(-1) + g1*(-1) + g2*(1)  = -g0 -g1 +g2
    p0 = -g0 - g1 - g2
    p1 =  g0 - g1 - g2
    p2 = -g0 + g1 - g2
    p3 = -g0 - g1 + g2

    # dot_wp = sum_k W_k * p_k
    dot_wp = w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3

    # softmax_grad_k = W_k * (p_k - dot_wp)
    sg0 = w0 * (p0 - dot_wp)
    sg1 = w1 * (p1 - dot_wp)
    sg2 = w2 * (p2 - dot_wp)
    sg3 = w3 * (p3 - dot_wp)

    # dL/d(x_c) = sum_k sg_k * 2(c_k[c] - x_c) / τ
    # = (2/τ) * [ sum_k sg_k * c_k[c]  -  x_c * sum_k sg_k ]
    # Note: sum_k sg_k = sum_k W_k*(p_k - dot_wp) = (sum_k W_k*p_k) - dot_wp*(sum_k W_k)
    #      = dot_wp - dot_wp = 0   (property of softmax backward)
    # So the x_c term vanishes!
    # dL/d(x_c) = (2/τ) * sum_k sg_k * c_k[c]

    two_inv_temp = 2.0 * inv_temp

    # For channel 0: c_k[0] = {-1, 1, -1, -1}
    grad_x0 = two_inv_temp * (-sg0 + sg1 - sg2 - sg3)
    # For channel 1: c_k[1] = {-1, -1, 1, -1}
    grad_x1 = two_inv_temp * (-sg0 - sg1 + sg2 - sg3)
    # For channel 2: c_k[2] = {-1, -1, -1, 1}
    grad_x2 = two_inv_temp * (-sg0 - sg1 - sg2 + sg3)

    tl.store(GRAD_IN_ptr + base + 0 * stride_c + h_vals * stride_h + w_vals * stride_w, grad_x0, mask=mask)
    tl.store(GRAD_IN_ptr + base + 1 * stride_c + h_vals * stride_h + w_vals * stride_w, grad_x1, mask=mask)
    tl.store(GRAD_IN_ptr + base + 2 * stride_c + h_vals * stride_h + w_vals * stride_w, grad_x2, mask=mask)


# ── Autograd wrapper ─────────────────────────────────────────────────
class _TritonColorQuantFn(torch.autograd.Function):
    """Fused forward + backward for color quantization using Triton."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        temperature: float,
        training: bool,
    ) -> torch.Tensor:
        # Ensure x is contiguous so that empty_like(x) creates an output buffer
        # with matching strides.  Without this, a non-contiguous slice such as
        # out_facie[:, :3, ...] of a 6-channel channels_last tensor (produced in
        # impedance mode) has strides (864,1,72,6) while empty_like gives only
        # 12960 elements with strides (432,1,36,3); the kernel then writes at
        # offsets up to 25917, causing silent out-of-bounds GPU memory corruption
        # that manifests as NaN gradients.
        if not x.is_contiguous():
            x = x.contiguous()

        B, _C, H, W = x.shape
        assert _C == 3, f"Expected 3 channels, got {_C}"
        HW = H * W
        inv_temp = 1.0 / temperature

        out = torch.empty_like(x)
        weights = torch.empty(
            (B, K, H, W), device=x.device, dtype=x.dtype
        ) if training else x.new_empty(0)

        BLOCK: int = min(1024, int(triton.next_power_of_2(HW)))  # type: ignore[arg-type]
        grid: tuple[int, ...] = (B * int(triton.cdiv(HW, BLOCK)),)  # type: ignore[arg-type]

        _color_quant_fwd_kernel[grid](  # type: ignore[index, arg-type]
            x, out, weights,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            HW, inv_temp, BLOCK, int(training),  # type: ignore[arg-type]
        )

        if training:
            ctx.save_for_backward(weights)
            ctx.inv_temp = inv_temp  # type: ignore[attr-defined]
            ctx.shape_info = (B, H, W, HW)  # type: ignore[attr-defined]
            # Remember the memory format so we can enforce it on grad_output
            # (autograd may pass expanded/broadcasted tensors with stride 0).
            ctx.mem_format = (  # type: ignore[attr-defined]
                torch.channels_last
                if x.is_contiguous(memory_format=torch.channels_last)
                else torch.contiguous_format
            )

        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, None, None]:
        (weights,) = ctx.saved_tensors  # type: ignore[attr-defined]
        weights_tensor: torch.Tensor = cast(torch.Tensor, weights)
        inv_temp: float = cast(float, ctx.inv_temp)  # type: ignore[attr-defined]
        B, H, W, HW = cast(tuple[int, int, int, int], ctx.shape_info)  # type: ignore[attr-defined]
        _ = H, W  # silence unused-variable error; shape_info stores (B, H, W, HW)
        mem_format: torch.memory_format = cast(torch.memory_format, ctx.mem_format)  # type: ignore[attr-defined]

        # Ensure grad_output has real strides (autograd can pass expanded
        # tensors with stride-0, e.g. from sum().backward()).
        grad_output = grad_output.contiguous(memory_format=mem_format)  # type: ignore[arg-type]
        stride_b, stride_c, stride_h, stride_w = grad_output.stride()

        grad_input = torch.empty_like(grad_output)

        BLOCK: int = min(1024, int(triton.next_power_of_2(HW)))  # type: ignore[arg-type]
        grid: tuple[int, ...] = (B * int(triton.cdiv(HW, BLOCK)),)  # type: ignore[arg-type]

        _color_quant_bwd_kernel[grid](  # type: ignore[index, arg-type]
            grad_output, weights_tensor, grad_input,
            stride_b, stride_c, stride_h, stride_w,
            HW, inv_temp, BLOCK,  # type: ignore[arg-type]
        )

        return grad_input, None, None


def triton_color_quantize(
    x: torch.Tensor, temperature: float, training: bool
) -> torch.Tensor:
    """Fused color quantization using Triton.

    Drop-in replacement for ``TorchColorQuantization.forward``.
    Performs squared-distance computation, softmax (K=4), and
    weighted-sum (C=3) in a single kernel launch.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape ``(B, 3, H, W)`` with values in [-1, 1].
    temperature : float
        Softmax temperature for soft assignment.
    training : bool
        Whether to use soft (True) or hard (False) quantization.

    Returns
    -------
    torch.Tensor
        Quantized tensor with same shape.
    """
    return _TritonColorQuantFn.apply(x, temperature, training)  # type: ignore[no-any-return]
