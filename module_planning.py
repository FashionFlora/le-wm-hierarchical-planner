import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


#####################################################################
##  Hierarchical Recurrent State Refiner                            ##
##                                                                  ##
##  Implements the "Recursive State Block" from the design notes:   ##
##  the world state is represented at three abstraction levels      ##
##  pooled from the encoder's patch tokens                          ##
##    Z_detail (n=15) : fine, local detail   ("co widzę")           ##
##    Z_rel    (n= 5) : relations            ("co to znaczy")       ##
##    Z_glob   (n= 1) : global / abstract    ("co z tego wynika")   ##
##  and these levels are *iteratively reconciled* for K cycles with ##
##  bottom-up then top-down GATED RESIDUAL cross-attention updates: ##
##      for _ in range(K):                                          ##
##          Z_rel    <- update_from_lower(Z_rel,    Z_detail)       ##
##          Z_glob   <- update_from_lower(Z_glob,   Z_rel)          ##
##          Z_rel    <- update_from_upper(Z_rel,    Z_glob)         ##
##          Z_detail <- update_from_upper(Z_detail, Z_rel)          ##
##  Top-down passes are corrections (residual), not generation:     ##
##      delta = CrossAttn(q=Z_detail, kv=Z_rel)                     ##
##      Z_detail = LayerNorm(Z_detail + gate * delta)               ##
##  K is "adaptive compute": K=0/refiner off == vanilla baseline,   ##
##  K=1/2/3 are the ablations described in the notes.               ##
#####################################################################


class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries from `x`, keys/values from `context`."""

    def __init__(self, dim, heads=3, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context):
        """
        x       : (B, Nq, D)  -- queries
        context : (B, Nk, D)  -- keys / values
        """
        q = self.to_q(self.norm_q(x))
        k, v = self.to_kv(self.norm_kv(context)).chunk(2, dim=-1)
        q, k, v = (rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in (q, k, v))
        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)  # non-causal
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CrossAttentionBlock(nn.Module):
    """Perceiver-style pooling block: learned queries attend over a context set.

    The query is kept as a residual so distinct learned query tokens stay
    distinct even when the context is a single token (the prediction case).
    """

    def __init__(self, dim, heads=3, dim_head=64, mlp_dim=None, dropout=0.0):
        super().__init__()
        self.attn = CrossAttention(dim, heads, dim_head, dropout)
        self.ff = FeedForward(dim, mlp_dim or 4 * dim, dropout)

    def forward(self, x, context):
        x = x + self.attn(x, context)
        x = x + self.ff(x)
        return x


class GatedCrossUpdate(nn.Module):
    """One reconciliation step: gated residual cross-attention update of `z`
    using information from `source` (a different abstraction level).

        delta = CrossAttn(q=z, kv=source)
        gate  = sigmoid(gate_net([mean(z), mean(source)]))   # per-channel, in (0,1)
        z     = LayerNorm(z + gate * delta)
        z     = z + FF(z)

    The gate lets the model decide how strong the correction should be:
    small when the level is already consistent, large when it must be
    pulled toward the other level (cf. PushT: well-placed T -> small gate).
    """

    def __init__(self, dim, heads=3, dim_head=64, mlp_dim=None, dropout=0.0):
        super().__init__()
        self.attn = CrossAttention(dim, heads, dim_head, dropout)
        self.norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim or 4 * dim, dropout)
        self.gate_net = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, z, source, diag=None, name=None):
        """
        z      : (B, Nq, D)  -- level being updated
        source : (B, Nk, D)  -- level providing the correction
        diag   : optional dict to record the correction magnitude / gate (metrics)
        """
        delta = self.attn(z, source)
        summary = torch.cat([z.mean(1, keepdim=True), source.mean(1, keepdim=True)], dim=-1)
        gate = torch.sigmoid(self.gate_net(summary))  # (B, 1, D), broadcast over tokens
        z = self.norm(z + gate * delta)
        z = z + self.ff(z)
        if diag is not None:
            diag[f"{name}_delta_norm"] = delta.norm(dim=-1).mean().detach()
            diag[f"{name}_gate_mean"] = gate.mean().detach()
        return z


class RecursiveStateBlock(nn.Module):
    """Iteratively reconcile the three abstraction levels for `num_cycles` (K)
    cycles of bottom-up then top-down gated residual updates.

    num_cycles == 0 returns the levels untouched (pure pooling, no recurrence).
    """

    def __init__(self, dim, heads=3, dim_head=64, mlp_dim=None, num_cycles=2, dropout=0.0):
        super().__init__()
        self.num_cycles = num_cycles
        mk = lambda: GatedCrossUpdate(dim, heads, dim_head, mlp_dim, dropout)
        self.bu_rel_from_detail = mk()  # bottom-up: Z_rel    <- Z_detail
        self.bu_glob_from_rel = mk()    # bottom-up: Z_glob   <- Z_rel
        self.td_rel_from_glob = mk()    # top-down:  Z_rel    <- Z_glob
        self.td_detail_from_rel = mk()  # top-down:  Z_detail <- Z_rel

    def forward(self, z_detail, z_rel, z_glob, diag=None):
        for _ in range(self.num_cycles):
            # bottom-up: details inform relations inform abstraction
            z_rel = self.bu_rel_from_detail(z_rel, z_detail, diag, "bu_rel")
            z_glob = self.bu_glob_from_rel(z_glob, z_rel, diag, "bu_glob")
            # top-down: abstraction corrects relations corrects details
            z_rel = self.td_rel_from_glob(z_rel, z_glob, diag, "td_rel")
            z_detail = self.td_detail_from_rel(z_detail, z_rel, diag, "td_detail")
        return z_detail, z_rel, z_glob


class RecursiveStateRefiner(nn.Module):
    """Optional, ablatable latent refiner wired around the world model.

    Pools a context set (ViT patch tokens when encoding observations, or the
    single predicted embedding when refining a rollout step) into three
    abstraction levels, reconciles them with a RecursiveStateBlock, and reads
    them back into a residual correction added to the input embedding.

    The readout is zero-initialised so the refiner is an exact identity at the
    start of training; the vanilla model is recovered by setting the refiner to
    null (or num_cycles=0 for the no-recurrence ablation).

    Note: `context_dim` must match the dim of whatever is passed as context.
    For this repo the ViT-tiny hidden size equals `embed_dim`, so the same
    refiner can pool encoder patch tokens (encode path) and predicted
    embeddings (predict path) without a dim mismatch.
    """

    def __init__(
        self,
        dim,
        context_dim=None,
        level_tokens=(15, 5, 1),
        num_cycles=2,
        heads=3,
        dim_head=64,
        mlp_dim=None,
        dropout=0.0,
        refine_predictions=True,
    ):
        super().__init__()
        context_dim = context_dim or dim
        self.refine_predictions = refine_predictions
        # when True, forward() records detached diagnostics in self.diagnostics
        # (off by default so the training hot path stays clean).
        self.collect_diagnostics = False
        self.diagnostics = {}
        n_detail, n_rel, n_glob = level_tokens

        self.context_proj = (
            nn.Linear(context_dim, dim) if context_dim != dim else nn.Identity()
        )

        self.q_detail = nn.Parameter(torch.randn(1, n_detail, dim) * 0.02)
        self.q_rel = nn.Parameter(torch.randn(1, n_rel, dim) * 0.02)
        self.q_glob = nn.Parameter(torch.randn(1, n_glob, dim) * 0.02)

        self.pool_detail = CrossAttentionBlock(dim, heads, dim_head, mlp_dim, dropout)
        self.pool_rel = CrossAttentionBlock(dim, heads, dim_head, mlp_dim, dropout)
        self.pool_glob = CrossAttentionBlock(dim, heads, dim_head, mlp_dim, dropout)

        self.rsb = RecursiveStateBlock(
            dim, heads, dim_head, mlp_dim, num_cycles, dropout
        )

        # readout: pooled summary of all levels -> residual correction.
        # zero-init final layer => identity refiner at initialisation.
        self.readout = nn.Sequential(nn.LayerNorm(3 * dim), nn.Linear(3 * dim, dim))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def _pool(self, context):
        B = context.size(0)
        ctx = self.context_proj(context)
        z_detail = self.pool_detail(self.q_detail.expand(B, -1, -1), ctx)
        z_rel = self.pool_rel(self.q_rel.expand(B, -1, -1), ctx)
        z_glob = self.pool_glob(self.q_glob.expand(B, -1, -1), ctx)
        return z_detail, z_rel, z_glob

    def levels(self, context):
        """Reconciled (z_detail, z_rel, z_glob) for a context, without the readout.

        Used by the rollout metrics to compare predicted vs target states at
        each abstraction level (z_glob == Z1, z_rel == Z5, z_detail == Z15).
        """
        z_detail, z_rel, z_glob = self._pool(context)
        return self.rsb(z_detail, z_rel, z_glob)

    @staticmethod
    def _token_diversity(z):
        """Mean pairwise cosine distance among tokens (0 == collapsed, higher == diverse)."""
        n = z.size(1)
        if n < 2:
            return z.new_zeros(())
        zn = F.normalize(z, dim=-1)
        sim = zn @ zn.transpose(1, 2)  # (B, n, n) cosine similarity
        off_mean = (sim.sum((1, 2)) - n) / (n * (n - 1))  # mean off-diagonal sim
        return (1.0 - off_mean).mean()

    def forward(self, context, residual_emb):
        """
        context      : (B, M, context_dim)  -- patch tokens or single embedding
        residual_emb : (B, dim)             -- embedding to be corrected
        returns      : (B, dim)             -- residual_emb + learned correction
        """
        z_detail, z_rel, z_glob = self._pool(context)

        diag = {} if self.collect_diagnostics else None
        z_detail, z_rel, z_glob = self.rsb(z_detail, z_rel, z_glob, diag=diag)

        summary = torch.cat(
            [z_detail.mean(1), z_rel.mean(1), z_glob.mean(1)], dim=-1
        )  # (B, 3*dim)

        if diag is not None:
            self.diagnostics = self._build_diagnostics(z_detail, z_rel, diag)

        return residual_emb + self.readout(summary)

    def _build_diagnostics(self, z_detail, z_rel, diag):
        d = {
            "token_diversity_z15": self._token_diversity(z_detail).detach(),
            "token_diversity_z5": self._token_diversity(z_rel).detach(),
        }
        # top-down passes are the corrections (Z5 <- Z1, Z15 <- Z5)
        if "td_rel_delta_norm" in diag:
            d["correction_norm_z5"] = diag["td_rel_delta_norm"]
            d["correction_norm_z15"] = diag["td_detail_delta_norm"]
        gates = [v for k, v in diag.items() if k.endswith("_gate_mean")]
        if gates:
            d["gate_mean"] = torch.stack(gates).mean()
        return d


class RecursiveStateEncoder(RecursiveStateRefiner):
    """Encode context tokens into the actual hierarchical world state.

    Unlike RecursiveStateRefiner, this does not collapse Z15/Z5/Z1 into a
    residual correction around a vanilla embedding. The levels remain the
    model's state; `emb` is only a compatibility readout for existing LeWM
    losses/evaluation code.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dim = self.q_detail.size(-1)
        self.z5_to_z15 = CrossAttention(dim, kwargs.get("heads", 3), kwargs.get("dim_head", 64), kwargs.get("dropout", 0.0))
        self.z1_to_z15 = CrossAttention(dim, kwargs.get("heads", 3), kwargs.get("dim_head", 64), kwargs.get("dropout", 0.0))
        self.fuse_gate = nn.Sequential(
            nn.LayerNorm(3 * dim),
            nn.Linear(3 * dim, 2 * dim),
        )
        self.fuse_norm = nn.LayerNorm(dim)
        self.readout = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def fuse_levels(self, z15, z5, z1):
        """Fuse levels without destroying them: Z5/Z1 correct Z15 through gates."""
        z5_to_15 = self.z5_to_z15(z15, z5)
        z1_to_15 = self.z1_to_z15(z15, z1)
        summary = torch.cat([z15.mean(1), z5.mean(1), z1.mean(1)], dim=-1)
        gate5, gate1 = torch.sigmoid(self.fuse_gate(summary)).chunk(2, dim=-1)
        fused_z15 = self.fuse_norm(
            z15 + gate5.unsqueeze(1) * z5_to_15 + gate1.unsqueeze(1) * z1_to_15
        )
        return fused_z15

    def readout_state(self, state):
        fused_z15 = state.get("fused_z15")
        if fused_z15 is None:
            fused_z15 = self.fuse_levels(state["z15"], state["z5"], state["z1"])
        return self.readout(fused_z15.mean(1))

    def forward(self, context):
        z15, z5, z1 = self._pool(context)
        diag = {} if self.collect_diagnostics else None
        z15, z5, z1 = self.rsb(z15, z5, z1, diag=diag)
        fused_z15 = self.fuse_levels(z15, z5, z1)
        state = {
            "z15": z15,
            "z5": z5,
            "z1": z1,
            "fused_z15": fused_z15,
        }
        state["emb"] = self.readout_state(state)
        if diag is not None:
            self.diagnostics = self._build_diagnostics(z15, z5, diag)
        return state


class HierarchicalStatePredictor(nn.Module):
    """Predict the next hierarchical world state with separate time scales.

    This follows the intended planner flow more closely than the earlier
    sequence-shaped predictor:

        long  Z1 history  -> global next state
        mid   Z5 history  + predicted Z1  -> relational next state
        short Z15 history + predicted Z5  -> detailed next state

    The output is one next state at each level: (B, 1, N_level, D).
    """

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        z1_history=None,
        z5_history=None,
        z15_history=None,
    ):
        super().__init__()
        dim = output_dim or input_dim
        self.z1_history = z1_history or num_frames
        self.z5_history = z5_history or max(1, num_frames // 2)
        self.z15_history = z15_history or max(1, num_frames // 4)

        self.z1_pos = nn.Parameter(torch.randn(1, self.z1_history, input_dim))
        self.z5_pos = nn.Parameter(torch.randn(1, self.z5_history, input_dim))
        self.z15_pos = nn.Parameter(torch.randn(1, self.z15_history, input_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.global_planner = Transformer(
            input_dim, hidden_dim, dim, depth, heads, dim_head, mlp_dim, dropout,
            block_class=ConditionalBlock,
        )
        self.relation_planner = Transformer(
            input_dim, hidden_dim, dim, depth, heads, dim_head, mlp_dim, dropout,
            block_class=ConditionalBlock,
        )
        self.detail_planner = Transformer(
            input_dim, hidden_dim, dim, depth, heads, dim_head, mlp_dim, dropout,
            block_class=ConditionalBlock,
        )

        self.z1_norm = nn.LayerNorm(dim)
        self.z5_norm = nn.LayerNorm(dim)
        self.z15_norm = nn.LayerNorm(dim)
        self.z5_from_z1 = GatedCrossUpdate(
            dim, heads=3, dim_head=dim_head, mlp_dim=4 * dim, dropout=dropout
        )
        self.z15_from_z5 = GatedCrossUpdate(
            dim, heads=3, dim_head=dim_head, mlp_dim=4 * dim, dropout=dropout
        )

    @staticmethod
    def _summary(z):
        return z.mean(dim=2)

    @staticmethod
    def _tail(x, length):
        return x[:, -min(length, x.size(1)):]

    def _with_pos(self, x, pos):
        t = x.size(1)
        return self.dropout(x + pos[:, -t:])

    def forward(self, state_history, act_emb):
        z15_hist = self._tail(state_history["z15"], self.z15_history)
        z5_hist = self._tail(state_history["z5"], self.z5_history)
        z1_hist = self._tail(state_history["z1"], self.z1_history)

        act_z1 = self._tail(act_emb, z1_hist.size(1))
        act_z5 = self._tail(act_emb, z5_hist.size(1))
        act_z15 = self._tail(act_emb, z15_hist.size(1))

        # Z1: long-horizon/global next-state proposal.
        z1_in = self._with_pos(self._summary(z1_hist), self.z1_pos)
        z1_plan = self.global_planner(z1_in, act_z1)[:, -1]
        pred_z1 = self.z1_norm(z1_hist[:, -1] + z1_plan.unsqueeze(1)).unsqueeze(1)

        # Z5: medium-horizon relational next-state, top-down conditioned by pred_z1.
        z1_cond = pred_z1[:, 0, 0].unsqueeze(1).expand(-1, z5_hist.size(1), -1)
        z5_in = self._with_pos(self._summary(z5_hist), self.z5_pos)
        z5_plan = self.relation_planner(z5_in, act_z5 + z1_cond)[:, -1]
        z5_seed = self.z5_from_z1(z5_hist[:, -1], pred_z1[:, 0])
        pred_z5 = self.z5_norm(z5_seed + z5_plan.unsqueeze(1)).unsqueeze(1)

        # Z15: short-horizon detailed next-state, top-down conditioned by pred_z5.
        z5_cond = pred_z5[:, 0].mean(1).unsqueeze(1).expand(-1, z15_hist.size(1), -1)
        z15_in = self._with_pos(self._summary(z15_hist), self.z15_pos)
        z15_plan = self.detail_planner(z15_in, act_z15 + z5_cond)[:, -1]
        z15_seed = self.z15_from_z5(z15_hist[:, -1], pred_z5[:, 0])
        pred_z15 = self.z15_norm(z15_seed + z15_plan.unsqueeze(1)).unsqueeze(1)

        return {"z15": pred_z15, "z5": pred_z5, "z1": pred_z1}


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x
