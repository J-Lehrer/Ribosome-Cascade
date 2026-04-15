# Ribosome-Cascade Track 2: Native Architecture
# ================================================
# Architecture Specification
#
# A transformer trained from scratch where the ribosome is a
# first-class architectural component, not a bolt-on.
#
# ARCHITECTURE OVERVIEW
# =====================
#
#   Input tokens
#       │
#       ▼
#   [Token Embedding + Positional Encoding]
#       │
#       ▼
#   ┌─────────────────────────────┐
#   │  LOWER TRANSFORMER (4 layers)  │  ← Standard token-level processing
#   │  Token-level self-attention     │  ← Builds contextual representations
#   └─────────────────────────────┘
#       │
#       ▼  (seq_len tokens, each with hidden_size features)
#   ┌─────────────────────────────┐
#   │  RIBOSOME LAYER              │  ← Scores, groups, compresses
#   │  1. Score importance per token│
#   │  2. Soft boundary detection   │
#   │  3. Chunk encoding (Perceiver)│
#   └─────────────────────────────┘
#       │
#       ▼  (n_chunks metatokens, each with hidden_size features)
#   ┌─────────────────────────────┐
#   │  CASCADE PROCESSOR (2 layers) │  ← Priority-ordered chunk processing
#   │  Causal attention by weight   │  ← Heaviest chunk = anchor
#   └─────────────────────────────┘
#       │
#       ▼
#   ┌─────────────────────────────┐
#   │  UPPER TRANSFORMER (4 layers) │  ← Chunk-level processing
#   │  Standard self-attention      │  ← But on metatokens, not tokens
#   └─────────────────────────────┘
#       │
#       ▼
#   ┌─────────────────────────────┐
#   │  CHUNK DECODER               │  ← Expand chunks back to tokens
#   │  Cross-attention: tokens      │
#   │  attend to processed chunks   │
#   └─────────────────────────────┘
#       │
#       ▼
#   [LM Head → vocab logits]
#
#
# PARAMETER BUDGET (~250M on RTX 5090 32GB)
# ==========================================
#   hidden_size = 768
#   vocab_size = 50257 (GPT-2 tokenizer)
#   max_seq_len = 1024
#   n_chunks = variable (learned boundaries, target ~seq_len/4)
#
#   Token embedding:     768 * 50257           = ~38.6M
#   Lower transformer:   4 layers * ~7M each   = ~28M
#   Ribosome layer:      scorer + boundary      = ~2M
#   Chunk encoder:       Perceiver cross-attn   = ~5M
#   Cascade processor:   2 layers               = ~14M
#   Upper transformer:   4 layers               = ~28M
#   Chunk decoder:       cross-attn + expand    = ~5M
#   LM head:             tied with embedding    = 0 (weight tying)
#   ─────────────────────────────────────────
#   Total:                                      ≈ 121M
#
#   Room to grow: could double layer counts or hidden_size
#
#
# KEY DESIGN DECISIONS
# ====================
#
# 1. DIFFERENTIABLE GROUPING: Gumbel-softmax boundary prediction
#    - Ribosome outputs boundary probability per token position
#    - During training: Gumbel-softmax samples soft boundaries
#    - During inference: hard argmax boundaries
#    - This avoids the two-phase training complexity
#
# 2. VARIABLE-LENGTH CHUNKS: Perceiver-style chunk encoder
#    - Predicts K boundary positions (K = target number of chunks)
#    - Tokens between boundaries form a chunk
#    - Each chunk compressed via cross-attention with a learnable query
#    - NOT mean-pooling — learned compression
#
# 3. PRIORITY CASCADE: causal attention by chunk weight
#    - Each chunk's weight = sum of token importance scores
#    - Chunks sorted by weight descending
#    - Causal mask: chunk i attends only to chunks 0..i-1
#    - Heaviest chunk (anchor) processed first, establishes context
#    - Lighter chunks conditioned on anchor
#
# 4. TRAINING DATA: OpenWebText (subset)
#    - wikitext-2 too small for from-scratch training
#    - OpenWebText ~38GB, use 1-2GB subset initially
#    - Can scale up as architecture validates
#
# 5. TWO-STAGE TRAINING:
#    - Stage 1 (warmup): Train without ribosome layer active
#      Lower + Upper transformers learn basic LM, ribosome scores
#      accumulate but don't affect forward pass. 5-10% of total steps.
#    - Stage 2 (full): Enable ribosome compression. The model must
#      now route through chunks. Gradual activation via alpha ramp.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# BUILDING BLOCKS
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())
    
    def forward(self, seq_len):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary(x, cos, sin):
    """Apply rotary embeddings. x: (B, n_heads, S, head_dim)"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    cos = cos[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, rope=None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rope = rope
    
    def forward(self, x, causal_mask=None):
        B, S, H = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each: (B, S, n_heads, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        if self.rope is not None:
            cos, sin = self.rope(S)
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)
        
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        if causal_mask is not None:
            attn = attn.masked_fill(causal_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, H)
        return self.o_proj(out)


class FFN(nn.Module):
    """SwiGLU feed-forward."""
    def __init__(self, hidden_size, ff_mult=4):
        super().__init__()
        ff_dim = int(hidden_size * ff_mult * 2 / 3)  # SwiGLU correction
        self.w1 = nn.Linear(hidden_size, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, ff_dim, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def sinusoidal_position_encoding(positions, dim):
    """Compute sinusoidal positional encoding at arbitrary (fractional) positions.
    
    Args:
        positions: (B, K) — fractional positions in sequence
        dim: hidden dimension (must be even)
    Returns:
        encoding: (B, K, dim) — positional encoding vectors
    """
    inv_freq = 1.0 / (10000 ** (
        torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim
    ))  # (dim/2,)
    # (B, K, 1) x (1, 1, dim/2) -> (B, K, dim/2)
    freqs = positions.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
    return torch.cat([freqs.sin(), freqs.cos()], dim=-1)  # (B, K, dim)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, n_heads, rope=None):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = SelfAttention(hidden_size, n_heads, rope)
        self.norm2 = RMSNorm(hidden_size)
        self.ffn = FFN(hidden_size)
    
    def forward(self, x, causal_mask=None):
        x = x + self.attn(self.norm1(x), causal_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# RIBOSOME LAYER
# ============================================================

class RibosomeLayer(nn.Module):
    """
    The core innovation: scores tokens, detects boundaries,
    compresses into metatokens.
    
    Boundary detection uses Gumbel-softmax for differentiable
    discrete boundary placement during training.
    """
    
    def __init__(self, hidden_size, max_chunks=64, n_heads=4, temperature=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_chunks = max_chunks
        
        # Importance scorer
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Boundary predictor: outputs probability of boundary at each position
        self.boundary_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )
        
        # Chunk encoder: learnable query per chunk slot
        self.chunk_query = nn.Parameter(torch.randn(1, max_chunks, hidden_size) * 0.02)
        self.chunk_cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads=n_heads, batch_first=True
        )
        self.chunk_norm = RMSNorm(hidden_size)
        
        # Temperature for Gumbel-softmax (annealed during training)
        self.gumbel_temperature = temperature
    
    def forward(self, hidden_states, return_debug=False, padding_mask=None):
        """
        Args:
            hidden_states: (B, S, H) from lower transformer
            padding_mask: (B, S) optional, True for real tokens, False for padding.
                         If None, all tokens are treated as real.
        Returns:
            chunk_repr: (B, K, H) compressed metatoken representations
            chunk_weights: (B, K) importance weight per chunk
            token_to_chunk: (B, S, K) soft assignment matrix
            importance: (B, S) per-token importance scores
        """
        B, S, H = hidden_states.shape
        K = min(self.max_chunks, S // 2)  # at most S/2 chunks
        
        # --- Score importance ---
        importance = self.scorer(hidden_states).squeeze(-1)  # (B, S)
        
        # Mask padding: zero importance and boundaries for padding tokens
        if padding_mask is not None:
            importance = importance * padding_mask.float()
        
        # --- Predict boundaries via Gumbel-softmax ---
        boundary_logits = self.boundary_predictor(hidden_states).squeeze(-1)  # (B, S)
        
        if self.training:
            # Gumbel-softmax: sample K boundary positions differentiably
            # Add Gumbel noise and take top-K via softmax
            gumbel_noise = -torch.log(-torch.log(
                torch.rand_like(boundary_logits) + 1e-8) + 1e-8)
            noisy_logits = (boundary_logits + gumbel_noise) / self.gumbel_temperature
        else:
            noisy_logits = boundary_logits
        
        # Soft boundary scores per position
        boundary_probs = torch.sigmoid(noisy_logits)  # (B, S)
        
        # Mask padding: zero boundary probs so padding doesn't create chunks
        if padding_mask is not None:
            boundary_probs = boundary_probs * padding_mask.float()
        
        # --- Build soft token-to-chunk assignment ---
        # Each token is assigned to the chunk defined by the nearest
        # boundary to its left. We use cumulative sum of boundary probs
        # as a soft "chunk index" and then compute assignment via
        # distance to each chunk slot.
        
        # Cumulative boundary creates a soft chunk index per token
        cum_boundaries = torch.cumsum(boundary_probs, dim=-1)  # (B, S)
        # Normalize to [0, K-1]
        max_cum = cum_boundaries[:, -1:].clamp(min=1.0)
        chunk_indices = cum_boundaries / max_cum * (K - 1)  # (B, S), values in [0, K-1]
        
        # Soft assignment: Gaussian kernel between token's chunk_index and each slot
        slots = torch.arange(K, device=hidden_states.device, dtype=torch.float32)
        slots = slots.unsqueeze(0).unsqueeze(0)  # (1, 1, K)
        chunk_indices_exp = chunk_indices.unsqueeze(-1)  # (B, S, 1)
        
        # Assignment weight: exp(-0.5 * (token_idx - slot)^2 / sigma^2)
        sigma = max(1.0, K / S * 2.0)  # adaptive width
        assign = torch.exp(-0.5 * ((chunk_indices_exp - slots) / sigma) ** 2)
        assign = assign / (assign.sum(dim=-1, keepdim=True) + 1e-8)  # normalize per token
        # assign: (B, S, K) — soft token-to-chunk assignment
        
        # Mask padding: zero assignment so padding tokens don't pollute chunks
        if padding_mask is not None:
            assign = assign * padding_mask.float().unsqueeze(-1)
        
        # --- Compress tokens into chunks ---
        # Weight assignment by importance (high-importance tokens dominate chunk repr)
        weighted_assign = assign * importance.unsqueeze(-1)  # (B, S, K)
        weighted_assign = weighted_assign / (weighted_assign.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weighted sum of hidden states per chunk
        chunk_repr = torch.bmm(weighted_assign.transpose(1, 2), hidden_states)  # (B, K, H)
        
        # Refine via cross-attention with learnable queries
        queries = self.chunk_query[:, :K, :].expand(B, -1, -1)
        refined, _ = self.chunk_cross_attn(queries, chunk_repr, chunk_repr)
        chunk_repr = self.chunk_norm(chunk_repr + refined)
        
        # --- Compute chunk weights (for cascade priority) ---
        # Each chunk's weight = sum of importance scores it captures
        chunk_weights = torch.bmm(
            importance.unsqueeze(1),  # (B, 1, S)
            assign  # (B, S, K)
        ).squeeze(1)  # (B, K)
        
        # --- Compute chunk positions (weighted mean of source token positions) ---
        positions = torch.arange(S, device=hidden_states.device, dtype=torch.float32)
        # assign: (B, S, K) -> transpose to (B, K, S), multiply by positions (S,)
        chunk_positions = torch.bmm(
            assign.transpose(1, 2),  # (B, K, S)
            positions.view(1, S, 1).expand(B, -1, -1)  # (B, S, 1)
        ).squeeze(-1)  # (B, K)
        # Normalize by total assignment mass per chunk
        chunk_mass = assign.sum(dim=1)  # (B, K)
        chunk_positions = chunk_positions / (chunk_mass + 1e-8)  # (B, K)
        
        if return_debug:
            return chunk_repr, chunk_weights, assign, importance, boundary_probs, chunk_positions
        return chunk_repr, chunk_weights, assign, importance, chunk_positions


# ============================================================
# CASCADE PROCESSOR
# ============================================================

class CascadeProcessor(nn.Module):
    """
    Processes chunks in priority order (heaviest first).
    Uses causal attention so each chunk only sees already-processed
    heavier chunks. The anchor (heaviest) establishes context;
    lighter chunks are interpreted through the anchor's lens.
    """
    
    def __init__(self, hidden_size, n_heads, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, n_heads)
            for _ in range(n_layers)
        ])
    
    def forward(self, chunk_repr, chunk_weights):
        """
        Args:
            chunk_repr: (B, K, H)
            chunk_weights: (B, K)
        Returns:
            processed: (B, K, H) — chunks processed in priority order
        """
        B, K, H = chunk_repr.shape
        
        # Sort by weight descending (heaviest first)
        sort_idx = chunk_weights.argsort(dim=-1, descending=True)
        sorted_repr = torch.gather(
            chunk_repr, 1, sort_idx.unsqueeze(-1).expand(-1, -1, H)
        )
        
        # Causal mask: chunk i can only attend to chunks 0..i
        causal = torch.triu(
            torch.ones(K, K, device=chunk_repr.device, dtype=torch.bool),
            diagonal=1
        )
        
        # Process through cascade layers
        x = sorted_repr
        for layer in self.layers:
            x = layer(x, causal_mask=causal)
        
        # Unsort back to original order
        unsort_idx = sort_idx.argsort(dim=-1)
        processed = torch.gather(
            x, 1, unsort_idx.unsqueeze(-1).expand(-1, -1, H)
        )
        
        return processed


# ============================================================
# CHUNK DECODER
# ============================================================

def _causal_token_to_chunk_mask(S, K, device, slack=1):
    """True-means-BLOCKED mask of shape (S, K) for token-to-chunk cross-attention.
    Token at position t may attend to chunks c where c <= floor(t*K/S) + slack.
    This enforces that a token-level output at position t cannot see information
    from chunks whose source positions exceed t (i.e., future tokens).
    """
    t_idx = torch.arange(S, device=device)
    c_idx = torch.arange(K, device=device)
    # allowed: c <= t*K//S + slack   =>   block when c > (t*K)//S + slack
    ceiling = (t_idx * K) // S + slack  # (S,)
    mask = c_idx.view(1, K) > ceiling.view(S, 1)  # (S, K)
    return mask  # True = block


class ChunkDecoder(nn.Module):
    """
    Expands processed chunk representations back to token-level
    predictions. Each token cross-attends to the processed chunks,
    weighted by its original assignment.

    If causal_chunks=True, the token-to-chunk cross-attention is masked so
    token t cannot attend to chunks whose source positions exceed t.
    """
    
    def __init__(self, hidden_size, n_heads, causal_chunks=False, chunk_mask_slack=1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads=n_heads, batch_first=True
        )
        self.norm = RMSNorm(hidden_size)
        self.ffn = FFN(hidden_size)
        self.ffn_norm = RMSNorm(hidden_size)
        self.causal_chunks = causal_chunks
        self.chunk_mask_slack = chunk_mask_slack
    
    def forward(self, token_states, chunk_repr, token_to_chunk):
        """
        Args:
            token_states: (B, S, H) — original token representations from lower transformer
            chunk_repr: (B, K, H) — processed chunk representations
            token_to_chunk: (B, S, K) — soft assignment matrix
        Returns:
            expanded: (B, S, H) — token-level representations
        """
        attn_mask = None
        if self.causal_chunks:
            S = token_states.size(1); K = chunk_repr.size(1)
            attn_mask = _causal_token_to_chunk_mask(
                S, K, token_states.device, slack=self.chunk_mask_slack)
        # Cross-attention: tokens query, chunks are keys/values
        decoded, _ = self.cross_attn(token_states, chunk_repr, chunk_repr,
                                     attn_mask=attn_mask, need_weights=False)
        x = self.norm(token_states + decoded)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ReverseRibosome(nn.Module):
    """
    The reverse ribosome: expands chunk-level representations back to
    full token-level resolution with enough capacity to recover
    fine-grained sequential detail.
    
    Pipeline:
      1. Cross-attention: each token attends to processed chunks
         (injects chunk-level context into token representations)
      2. Residual add with original token_states
         (preserves fine-grained token identity from embed layers)
      3. N causal self-attention layers at full token resolution
         (recovers sequential dependencies lost in compression)
    
    The self-attention layers are the key: they operate at full S
    tokens with causal masking, so they can reconstruct the
    fine-grained token-by-token predictions that compression destroyed.
    """
    
    def __init__(self, hidden_size, n_heads, n_layers=2, rope=None,
                 causal_chunks=False, chunk_mask_slack=1):
        super().__init__()
        # Step 1: cross-attention from tokens to chunks
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads=n_heads, batch_first=True
        )
        self.cross_norm = RMSNorm(hidden_size)
        self.cross_ffn = FFN(hidden_size)
        self.cross_ffn_norm = RMSNorm(hidden_size)
        
        # Step 2: token-level self-attention layers (with RoPE for position)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, n_heads, rope)
            for _ in range(n_layers)
        ])
        self.out_norm = RMSNorm(hidden_size)
        self.causal_chunks = causal_chunks
        self.chunk_mask_slack = chunk_mask_slack
    
    def forward(self, token_states, chunk_repr, token_to_chunk):
        """
        Args:
            token_states: (B, S, H) — from embed layers (has positional info)
            chunk_repr: (B, K, H) — processed chunk representations
            token_to_chunk: (B, S, K) — soft assignment matrix
        Returns:
            output: (B, S, H) — refined token-level representations
        """
        B, S, H = token_states.shape
        K = chunk_repr.size(1)
        
        # Cross-attention: inject chunk context into token representations.
        # If causal_chunks, token t can only attend to chunks covering positions <= t.
        cross_mask = None
        if self.causal_chunks:
            cross_mask = _causal_token_to_chunk_mask(
                S, K, token_states.device, slack=self.chunk_mask_slack)
        decoded, _ = self.cross_attn(token_states, chunk_repr, chunk_repr,
                                     attn_mask=cross_mask, need_weights=False)
        x = self.cross_norm(token_states + decoded)
        x = x + self.cross_ffn(self.cross_ffn_norm(x))
        
        # Causal self-attention at full token resolution
        causal = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
        )
        for layer in self.layers:
            x = layer(x, causal_mask=causal)
        
        return self.out_norm(x)


# ============================================================
# FULL MODEL
# ============================================================

class RibosomeCascadeNative(nn.Module):
    """
    Full native Ribosome-Cascade transformer.
    
    Architecture:
        Embedding → Lower Transformer (4L) → Ribosome Layer →
        Cascade Processor (2L) → Upper Transformer (4L) →
        Chunk Decoder → LM Head
    """
    
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        n_heads=12,
        lower_layers=4,
        upper_layers=4,
        cascade_layers=2,
        max_seq_len=1024,
        max_chunks=64,
        dropout=0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        
        # Token embedding (weight-tied with LM head)
        self.tok_emb = nn.Embedding(vocab_size, hidden_size)
        
        # Rotary embeddings
        self.rope = RotaryEmbedding(hidden_size // n_heads, max_seq_len)
        
        # Lower transformer: token-level processing
        self.lower = nn.ModuleList([
            TransformerBlock(hidden_size, n_heads, self.rope)
            for _ in range(lower_layers)
        ])
        self.lower_norm = RMSNorm(hidden_size)
        
        # Ribosome layer: score + group + compress
        self.ribosome = RibosomeLayer(hidden_size, max_chunks, n_heads)
        
        # Cascade processor: priority-ordered chunk attention
        self.cascade = CascadeProcessor(hidden_size, n_heads, cascade_layers)
        
        # Upper transformer: chunk-level processing
        self.upper = nn.ModuleList([
            TransformerBlock(hidden_size, n_heads)
            for _ in range(upper_layers)
        ])
        self.upper_norm = RMSNorm(hidden_size)
        
        # Chunk decoder: expand back to tokens
        self.decoder = ChunkDecoder(hidden_size, n_heads)
        
        # LM head (tied weights with embedding)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        
        # Training control
        self.ribosome_alpha = 1.0  # 0→1 ramp during training
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, input_ids, labels=None):
        B, S = input_ids.shape
        
        # Embed
        x = self.tok_emb(input_ids)  # (B, S, H)
        
        # Causal mask for token-level attention
        causal = torch.triu(
            torch.ones(S, S, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        
        # Lower transformer
        for layer in self.lower:
            x = layer(x, causal_mask=causal)
        token_states = self.lower_norm(x)  # save for decoder
        
        # Ribosome: compress to chunks
        chunk_repr, chunk_weights, assign, importance, chunk_positions = self.ribosome(token_states)
        
        # Inject sequence position metadata into chunks
        chunk_repr = chunk_repr + sinusoidal_position_encoding(
            chunk_positions, self.hidden_size)
        
        # Cascade: priority-ordered processing
        chunk_repr = self.cascade(chunk_repr, chunk_weights)
        
        # Upper transformer: chunk-level
        for layer in self.upper:
            chunk_repr = layer(chunk_repr)
        chunk_repr = self.upper_norm(chunk_repr)
        
        # Decode: expand chunks back to token-level
        decoded = self.decoder(token_states, chunk_repr, assign)
        
        # Blend with bypass (alpha ramp for training stability)
        output = self.ribosome_alpha * decoded + (1 - self.ribosome_alpha) * token_states
        
        # LM head
        logits = self.lm_head(output)
        
        loss = None
        if labels is not None:
            # No shift: loader already provides aligned input/label pairs
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            # Sparsity regularization: very gentle, let model learn importance first
            sparsity_loss = importance.mean()
            # Boundary regularization: encourage ~S/4 boundaries
            target_boundaries = S / 4
            boundary_count = assign.sum(dim=1).max(dim=-1).values.mean()
            boundary_loss = (boundary_count - target_boundaries).pow(2) / target_boundaries
            
            loss = ce_loss + 0.001 * sparsity_loss + 0.0001 * boundary_loss
        
        return loss, logits, importance
    
    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ============================================================
# QUICK VALIDATION
# ============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = RibosomeCascadeNative(
        vocab_size=50257,
        hidden_size=768,
        n_heads=12,
        lower_layers=4,
        upper_layers=4,
        cascade_layers=2,
        max_seq_len=1024,
        max_chunks=64,
    ).to(device)
    
    total, trainable = model.count_params()
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    
    # Smoke test
    x = torch.randint(0, 50257, (2, 128)).to(device)
    labels = x.clone()
    
    loss, logits, importance = model(x, labels)
    print(f"Loss: {loss.item():.4f}")
    print(f"Logits shape: {logits.shape}")
    print(f"Importance shape: {importance.shape}")
    print(f"Importance range: [{importance.min().item():.4f}, {importance.max().item():.4f}]")
    
    loss.backward()
    print("Backward pass OK")
    print(f"GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
