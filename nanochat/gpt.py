"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch._dynamo as dynamo
except Exception:
    dynamo = None

try:
    from nanochat.common import get_dist_info, print0
except ImportError:
    import os

    def print0(s="", **kwargs):
        ddp_rank = int(os.environ.get("RANK", 0))
        if ddp_rank == 0:
            print(s, **kwargs)

    def get_dist_info():
        if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
            return True, int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
        return False, 0, 0, 1
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
_MAX_LOGITS_BYTES = int(os.environ.get("NANOCHAT_MAX_LOGITS_BYTES", 1024 * 1024 * 1024))

def _sliding_causal_attn_mask(q_len: int, k_len: int, *, q_pos0: int, window_size, device):
    """
    Build an additive attention mask (float) where masked positions are -inf.
    q_pos0 is the absolute position of query index 0 within the full sequence.
    """
    left, right = window_size
    assert right == 0, "Only causal attention is supported (window_size right must be 0)"
    q_idx = torch.arange(q_len, device=device).view(q_len, 1)
    k_idx = torch.arange(k_len, device=device).view(1, k_len)
    q_abs = q_pos0 + q_idx
    # causal: keys cannot be in the future relative to each query position
    allowed = k_idx <= q_abs
    if left >= 0:
        # sliding window: allow only the last `left` tokens before current position (plus itself)
        allowed = allowed & (k_idx >= (q_abs - left))
    mask = torch.zeros((q_len, k_len), dtype=torch.float32, device=device)
    mask = mask.masked_fill(~allowed, float("-inf"))
    return mask


class _TorchSDPAFlashAttnFallback:
    """
    Minimal Flash Attention 3 interface implemented via PyTorch SDPA.
    This exists to keep training/inference usable when `kernels`/FA3 isn't available.
    """

    @staticmethod
    def _expand_kv_for_gqa(x: torch.Tensor, n_query_heads: int) -> torch.Tensor:
        # x: (B, T, Hkv, D) -> (B, T, Hq, D)
        b, t, h_kv, d = x.shape
        if h_kv == n_query_heads:
            return x
        assert n_query_heads % h_kv == 0, "n_heads must be a multiple of n_kv_heads for GQA"
        repeat = n_query_heads // h_kv
        return x.repeat_interleave(repeat, dim=2)

    def flash_attn_func(self, q, k, v, *, causal: bool, window_size):
        assert causal is True, "Only causal attention is supported"
        b, t, hq, d = q.shape
        k = self._expand_kv_for_gqa(k, hq)
        v = self._expand_kv_for_gqa(v, hq)
        # SDPA uses (B, H, T, D)
        qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        left, right = window_size
        if left < 0 or left >= t:
            y = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
        else:
            mask = _sliding_causal_attn_mask(t, t, q_pos0=0, window_size=window_size, device=qh.device)
            y = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
        return y.transpose(1, 2)

    def flash_attn_with_kvcache(
        self,
        q,
        k_cache,
        v_cache,
        *,
        k,
        v,
        cache_seqlens,
        causal: bool,
        window_size,
    ):
        assert causal is True, "Only causal attention is supported"
        b, t, hq, d = q.shape
        pos0 = cache_seqlens.to(torch.int64)
        max_seq = k_cache.size(1)
        if (pos0 + t).max().item() > max_seq:
            raise ValueError(f"KV cache overflow: pos+t exceeds max_seq_len ({max_seq})")

        # Write new keys/values into the cache at each batch element's position.
        # Note: this is intentionally simple and correct; it's not optimized.
        for bi in range(b):
            p = int(pos0[bi].item())
            k_cache[bi, p:p+t, :, :] = k[bi]
            v_cache[bi, p:p+t, :, :] = v[bi]

        # Attend over the cache up to (pos0 + t) for each batch element.
        ys = []
        for bi in range(b):
            p = int(pos0[bi].item())
            k_len = p + t
            k_all = k_cache[bi : bi + 1, :k_len, :, :]
            v_all = v_cache[bi : bi + 1, :k_len, :, :]
            k_all = self._expand_kv_for_gqa(k_all, hq)
            v_all = self._expand_kv_for_gqa(v_all, hq)
            qh = q[bi : bi + 1].transpose(1, 2)
            kh = k_all.transpose(1, 2)
            vh = v_all.transpose(1, 2)
            left, right = window_size
            if p == 0 and (left < 0 or left >= t) and k_len == t:
                y = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
            else:
                mask = _sliding_causal_attn_mask(t, k_len, q_pos0=p, window_size=window_size, device=qh.device)
                y = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
            ys.append(y.transpose(1, 2))
        return torch.cat(ys, dim=0)


_FLASH_ATTN = None
_FLASH_ATTN_FALLBACK_WARNED = False


def _get_flash_attn():
    global _FLASH_ATTN, _FLASH_ATTN_FALLBACK_WARNED
    if _FLASH_ATTN is not None:
        return _FLASH_ATTN
    if os.environ.get("NANOCHAT_DISABLE_FA3", "") == "1":
        _FLASH_ATTN = _TorchSDPAFlashAttnFallback()
        return _FLASH_ATTN
    try:
        # Load Flash Attention 3 from HuggingFace Hub.
        # Official docs of FA3 label it as "beta" and want you to install FA3 from source, which is a pain.
        # Wishing for official FA3 wheels soon, for now this seems to be a fast way to get them (ty varunneal)
        from kernels import get_kernel  # type: ignore

        _FLASH_ATTN = get_kernel("varunneal/flash-attention-3").flash_attn_interface
        return _FLASH_ATTN
    except Exception as e:
        _FLASH_ATTN = _TorchSDPAFlashAttnFallback()
        if not _FLASH_ATTN_FALLBACK_WARNED:
            _FLASH_ATTN_FALLBACK_WARNED = True
            extra_hint = ""
            msg = str(e)
            if "GLIBC_" in msg and "not found" in msg:
                extra_hint = (
                    " Detected a GLIBC mismatch while loading the FA3 extension. "
                    "This usually means the downloaded binary was built against a newer OS (e.g. Ubuntu 22.04, glibc>=2.32) "
                    "than your runtime (e.g. Ubuntu 20.04, glibc 2.31). "
                    "Fix: use an Ubuntu 22.04-based container/image, or build/install Flash Attention 3 from source on the machine."
                )
            print0(
                "Flash Attention 3 unavailable; falling back to PyTorch SDPA. "
                "For best performance, install `kernels` and ensure HF Hub access. "
                f"(Reason: {type(e).__name__}: {e}){extra_hint} "
                "To silence this message, set NANOCHAT_DISABLE_FA3=1."
            )
        return _FLASH_ATTN

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "L"


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        # Attention with Flash Attention 3
        # FA3 handles GQA automatically when n_kv_heads < n_heads
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        flash_attn = _get_flash_attn()
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        with torch.no_grad():
            self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
            self.x0_lambdas.fill_(0.0)      # 0.0 => skip connection to input is disabled at init

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast token embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        nparams_exclude = self.transformer.wte.weight.numel() + self.resid_lambdas.numel() + self.x0_lambdas.numel()
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return all of the parameters, same as Chinchilla paper.
        Kaplan et al. did not include embedding parameters and said that this led to cleaner scaling laws.
        But Kaplan et al. also had a bug in their results (as pointed out by Chinchilla).
        My own experiments in nanochat confirm the Chinchilla approach gives the much cleaner scaling law.
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper <- good).
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper <- bad)
        """
        nparams = sum(p.numel() for p in self.parameters())
        return nparams

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 5 groups (matrix, embedding, lm_head, resid_lambdas, x0_lambdas)
        matrix_params_all = list(self.transformer.h.parameters())
        embedding_params_all = list(self.transformer.wte.parameters())
        lm_head_params_all = list(self.lm_head.parameters())
        resid_params_all = [self.resid_lambdas]
        x0_params_all = [self.x0_lambdas]
        assert len(list(self.parameters())) == len(matrix_params_all) + len(embedding_params_all) + len(lm_head_params_all) + len(resid_params_all) + len(x0_params_all)
        # Filter to trainable params so freezing is safe (Muon/AdamW expect grads).
        matrix_params = [p for p in matrix_params_all if p.requires_grad]
        embedding_params = [p for p in embedding_params_all if p.requires_grad]
        lm_head_params = [p for p in lm_head_params_all if p.requires_grad]
        resid_params = [p for p in resid_params_all if p.requires_grad]
        x0_params = [p for p in x0_params_all if p.requires_grad]
        # Create the AdamW optimizer for the embedding, lm_head, and per-layer scalars
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = []
        if lm_head_params:
            adam_groups.append(dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale))
        if embedding_params:
            adam_groups.append(dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale))
        if resid_params:
            adam_groups.append(dict(params=resid_params, lr=scalar_lr * 0.01)) # sensitive, accumulates in residual stream
        if x0_params:
            adam_groups.append(dict(params=x0_params, lr=scalar_lr))
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0) # NOTE: weight decay is hardcoded to 0.0 for AdamW, only used in Muon
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs) if adam_groups else None
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs) if matrix_params else None
        # Combine them the two optimizers into one list
        optimizers = [opt for opt in (adamw_optimizer, muon_optimizer) if opt is not None]
        if not optimizers:
            raise ValueError("No trainable parameters found; all params are frozen.")
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def _loss_from_hidden(self, x, targets, loss_reduction, softcap):
        if loss_reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported loss_reduction: {loss_reduction}")
        vocab_size = self.config.vocab_size
        b, t, c = x.size()
        flat_tokens = b * t
        targets_flat = targets.view(-1)
        total_logits_bytes = flat_tokens * vocab_size * 4  # float32 for loss/softcap
        if total_logits_bytes <= _MAX_LOGITS_BYTES:
            logits = self.lm_head(x)
            logits = logits[..., :vocab_size]
            logits = logits.float()
            logits = softcap * torch.tanh(logits / softcap)
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets_flat,
                ignore_index=-1,
                reduction=loss_reduction,
            )

        chunk_tokens = max(1, _MAX_LOGITS_BYTES // (vocab_size * 4))
        x_flat = x.view(flat_tokens, c)
        if loss_reduction == "none":
            losses = torch.empty(flat_tokens, dtype=torch.float32, device=x.device)
            for start in range(0, flat_tokens, chunk_tokens):
                end = min(start + chunk_tokens, flat_tokens)
                logits = self.lm_head(x_flat[start:end])
                logits = logits[..., :vocab_size]
                logits = logits.float()
                logits = softcap * torch.tanh(logits / softcap)
                losses[start:end] = F.cross_entropy(
                    logits,
                    targets_flat[start:end],
                    ignore_index=-1,
                    reduction="none",
                )
            return losses

        loss_sum = torch.zeros((), dtype=torch.float32, device=x.device)
        valid_tokens = torch.zeros((), dtype=torch.int64, device=x.device)
        for start in range(0, flat_tokens, chunk_tokens):
            end = min(start + chunk_tokens, flat_tokens)
            logits = self.lm_head(x_flat[start:end])
            logits = logits[..., :vocab_size]
            logits = logits.float()
            logits = softcap * torch.tanh(logits / softcap)
            loss_sum = loss_sum + F.cross_entropy(
                logits,
                targets_flat[start:end],
                ignore_index=-1,
                reduction="sum",
            )
            if loss_reduction == "mean":
                valid_tokens = valid_tokens + (targets_flat[start:end] != -1).sum()
        if loss_reduction == "sum":
            return loss_sum
        denom = valid_tokens.clamp_min(1).to(loss_sum.dtype)
        return loss_sum / denom

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', logits_positions=None):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, cos_sin, self.window_sizes[i], kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        # NOTE: (B, T, V) is extremely large when vocab_size is big (e.g. 262k).
        # For inference/evaluation that only needs logits at specific positions, pass logits_positions
        # to compute a (B, V) tensor instead and save a lot of memory.
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        if targets is not None:
            if logits_positions is not None:
                raise ValueError("logits_positions is not supported when targets is provided")
            loss_fn = self._loss_from_hidden
            if loss_reduction == "none" and dynamo is not None:
                loss_fn = dynamo.disable(loss_fn)
            return loss_fn(x, targets, loss_reduction, softcap)
        if logits_positions is not None:
            if isinstance(logits_positions, int):
                pos = torch.full((B,), logits_positions, dtype=torch.long, device=idx.device)
            else:
                pos = logits_positions
                if not torch.is_tensor(pos):
                    pos = torch.tensor(pos, dtype=torch.long, device=idx.device)
                pos = pos.to(device=idx.device, dtype=torch.long)
                if pos.ndim == 0:
                    pos = pos.expand(B)
            assert pos.shape == (B,), f"logits_positions must be shape (B,), got {tuple(pos.shape)}"
            assert (pos >= 0).all() and (pos < T).all(), "logits_positions out of range"
            x_sel = x[torch.arange(B, device=idx.device), pos]  # (B, C)
            logits = self.lm_head(x_sel)  # (B, padded_vocab_size)
            logits = logits[..., :self.config.vocab_size]  # slice to remove padding
        else:
            logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
            logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        # inference: just return the logits directly
        return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids, logits_positions=ids.size(1) - 1) # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
