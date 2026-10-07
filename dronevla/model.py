"""Tiny RGB + text behaviour-cloning policy (roadmap v3 §3.1, Phase 1 item 4).

    RGB 3x96x128 --CNN--> 6x8x64 map --flatten--> 128   \
    instruction --train vocabulary--> embeddings --mean--> 32   > concat -> MLP -> 5
    proprio (11, train mean/std) --MLP--> 32                    /

Outputs: 4 motion values in [-1, 1] (tanh), scaled by the §4.3 caps to (vx, vy, vz,
yaw_rate) in m/s and rad/s -- the "fixed scaling by the action cap" of §4.3 -- and one Stop
logit. The conv stack keeps a 6x8 spatial map before flattening, so where a colour sits in
the image (left or right) survives into the features.

Inputs are only what §4.3 lists: no target id, coordinates, layout id or template id. The
text embedding is computed once per episode (§3.1: cached at task start).
"""
from __future__ import annotations

import re

import torch
from torch import nn

PAD, UNK = "<pad>", "<unk>"
MAX_TOKENS = 32                      # §4.3 design value


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


class Vocab:
    """Train-only vocabulary (§4.2 item 6: a train-only tokenizer with <unk>)."""

    def __init__(self, tokens: list[str]):
        self.itos = [PAD, UNK] + sorted(set(tokens) - {PAD, UNK})
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def from_texts(cls, texts):
        return cls([tok for t in texts for tok in tokenize(t)])

    def encode(self, text: str) -> list[int]:
        toks = tokenize(text)
        if len(toks) > MAX_TOKENS:
            # §4.3: truncation is detected and rejected, never silently applied
            raise ValueError(f"instruction has {len(toks)} tokens, limit {MAX_TOKENS}: {text!r}")
        return [self.stoi.get(t, 1) for t in toks] or [1]

    def unknown_rate(self, text: str) -> float:
        toks = tokenize(text)
        return sum(t not in self.stoi for t in toks) / max(len(toks), 1)

    def __len__(self):
        return len(self.itos)


class TinyBC(nn.Module):
    """`film=True` adds FiLM conditioning: the instruction scales and shifts every conv
    layer's channels (x -> (1 + gamma) * x + beta, before the ReLU), so the words can change
    *what the CNN extracts* instead of only joining a summary at the end. The FiLM generator
    is zero-initialised, so training starts from exactly the unconditioned network.

    `aux_offsets=True` adds an auxiliary head on the fused features that predicts the
    body-frame offset to each colour's hover point (the world model's supervised target, see
    dronevla.world_model). It is a training signal only: the policy's action does not use it.
    It exists so the BC-vs-world-model comparison gives both the same privileged supervision.

    `attn=True` replaces the flatten-to-128 image path with language-conditioned
    cross-attention pooling: the 6x8 map becomes 48 patch tokens (plus a learned position
    embedding), the sentence vector becomes the query, and the attention-weighted sum of the
    patches is the only image information the head sees. The words choose *which* part of the
    image to look at; the patch features and their positions say *where* it is. The last
    attention weights are kept in `self.last_attn` (B, heads, 48) for inspection."""

    CHANNELS = (16, 32, 64, 64)

    ATTN_DIM, ATTN_HEADS, PATCHES = 256, 4, 6 * 8      # 256: parameter count within 80-100% of the plain model

    # Action tokens (OpenVLA-style discretisation): only vx and vy are tokenised; this task is
    # planar, so vz and yaw rate are always zero and the adapter forces them to zero anyway.
    TOKEN_DIMS = (0, 1)

    def __init__(self, vocab_size: int, text_dim: int = 32, proprio_dim: int = 11,
                 film: bool = False, aux_offsets: bool = False, attn: bool = False,
                 action_bins: int = 0, img_dim: int = 128, head_hidden: int = 128,
                 chunk: int = 1, paired_goal: bool = False):
        super().__init__()
        if film and attn:
            raise ValueError("film and attn are separate variants; pick one")
        if chunk > 1 and not action_bins:
            raise ValueError("action chunks are implemented for the token head only")
        if paired_goal and (film or attn):
            raise ValueError("paired_goal reuses one image feature for two sentences; "
                             "film/attn make the image path sentence-dependent")
        self.action_bins = action_bins
        self.chunk = chunk
        if action_bins:
            # bin edges per tokenised dimension, in cap-normalised units; set from the
            # training actions by `set_action_bins` and saved with the state dict
            self.register_buffer("bin_edges", torch.zeros(len(self.TOKEN_DIMS), action_bins + 1))
        c = self.CHANNELS
        self.convs = nn.ModuleList([
            nn.Conv2d(3, c[0], 5, stride=2, padding=2),       # 48 x 64
            nn.Conv2d(c[0], c[1], 3, stride=2, padding=1),    # 24 x 32
            nn.Conv2d(c[1], c[2], 3, stride=2, padding=1),    # 12 x 16
            nn.Conv2d(c[2], c[3], 3, stride=2, padding=1),    #  6 x  8
        ])
        self.film = nn.Linear(text_dim, 2 * sum(c)) if film else None
        if film:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        self.attn = attn
        self.last_attn = None
        if attn:
            d = self.ATTN_DIM
            self.patch_proj = nn.Linear(c[3], d)
            self.pos = nn.Parameter(torch.zeros(1, self.PATCHES, d))
            nn.init.normal_(self.pos, std=0.02)
            self.query = nn.Linear(text_dim, d)
            self.xattn = nn.MultiheadAttention(d, self.ATTN_HEADS, batch_first=True)
            self.img_fc = nn.Sequential(nn.Linear(d, img_dim), nn.ReLU())
        else:
            self.img_fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 6 * 8, img_dim), nn.ReLU())
        self.embed = nn.Embedding(vocab_size, text_dim, padding_idx=0)
        self.prop = nn.Sequential(nn.Linear(proprio_dim, 32), nn.ReLU())
        fused = img_dim + text_dim + 32
        n_out = len(self.TOKEN_DIMS) * action_bins + 1 if action_bins else 5
        if chunk > 1:
            # one trunk, a learned embedding per future step, one shared token decoder:
            # (B, fused) -> (B, H, D*bins + 1); the parameter count does not grow with H
            self.head_trunk = nn.Sequential(nn.Linear(fused, head_hidden), nn.ReLU())
            self.step_emb = nn.Parameter(torch.zeros(chunk, head_hidden))
            nn.init.normal_(self.step_emb, std=0.02)
            self.head_out = nn.Linear(head_hidden, n_out)
        else:
            self.head = nn.Sequential(nn.Linear(fused, head_hidden), nn.ReLU(),
                                      nn.Linear(head_hidden, n_out))
        self.aux_head = (nn.Sequential(nn.Linear(fused, 64), nn.ReLU(),
                                       nn.Linear(64, 8)) if aux_offsets else None)
        # paired goal head: offset (units of 3 m) to the hover point of the target the
        # sentence names; trained with both sentences of a pair on the same frame
        self.goal_head = (nn.Sequential(nn.Linear(fused, 64), nn.ReLU(), nn.Linear(64, 2))
                          if paired_goal else None)

    # ------------------------------------------------------------------ action tokens
    def set_action_bins(self, actions_norm: torch.Tensor):
        """Per tokenised dimension, `action_bins` uniform bins between the 1st and 99th
        percentile of the training actions (cap-normalised), as OpenVLA does; outliers fall
        into the end bins."""
        edges = []
        for d in self.TOKEN_DIMS:
            lo, hi = torch.quantile(actions_norm[:, d].float(), torch.tensor([0.01, 0.99])).tolist()
            if hi - lo < 1e-6:
                lo, hi = lo - 1e-3, hi + 1e-3
            edges.append(torch.linspace(lo, hi, self.action_bins + 1))
        self.bin_edges.copy_(torch.stack(edges))

    def bin_centres(self) -> torch.Tensor:
        return (self.bin_edges[:, :-1] + self.bin_edges[:, 1:]) / 2      # (D, bins)

    def tokenize_actions(self, actions_norm: torch.Tensor) -> torch.Tensor:
        """(B, 4) cap-normalised actions -> (B, D) bin indices in [0, bins-1]."""
        toks = []
        for i, d in enumerate(self.TOKEN_DIMS):
            inner = self.bin_edges[i, 1:-1].contiguous()
            toks.append(torch.bucketize(actions_norm[:, d].contiguous(), inner))
        return torch.stack(toks, dim=1)

    def detokenize(self, logits: torch.Tensor) -> torch.Tensor:
        """(B, D, bins) logits -> (B, 4) cap-normalised actions, greedy (argmax) decoding."""
        centres = self.bin_centres()
        idx = logits.argmax(-1)                                           # (B, D)
        out = torch.zeros(logits.shape[0], 4, dtype=centres.dtype, device=logits.device)
        for i, d in enumerate(self.TOKEN_DIMS):
            out[:, d] = centres[i][idx[:, i]]
        return out

    def cnn(self, x, text_vec=None):
        params = self.film(text_vec) if self.film is not None else None
        offset = 0
        for conv, ch in zip(self.convs, self.CHANNELS):
            x = conv(x)
            if params is not None:
                gamma = params[:, offset:offset + ch, None, None]
                beta = params[:, offset + ch:offset + 2 * ch, None, None]
                x = (1 + gamma) * x + beta
                offset += 2 * ch
            x = torch.relu(x)
        return x

    def attend(self, fmap: torch.Tensor, text_vec: torch.Tensor) -> torch.Tensor:
        """(B, 64, 6, 8) map + (B, text_dim) sentence -> (B, ATTN_DIM) attended patch summary."""
        tokens = self.patch_proj(fmap.flatten(2).transpose(1, 2)) + self.pos    # (B, 48, d)
        q = self.query(text_vec).unsqueeze(1)                                   # (B, 1, d)
        out, w = self.xattn(q, tokens, tokens, need_weights=True,
                            average_attn_weights=False)                         # w: (B, H, 1, 48)
        self.last_attn = w.squeeze(2).detach()
        return out.squeeze(1)

    def encode_text(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, T) token ids, 0 = pad -> (B, text_dim) mean over real tokens."""
        mask = (ids != 0).unsqueeze(-1).float()
        return (self.embed(ids) * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def forward(self, rgb_u8: torch.Tensor, proprio: torch.Tensor, text_vec: torch.Tensor,
                return_aux: bool = False, return_logits: bool = False):
        """rgb_u8 (B, 96, 128, 3) uint8; proprio (B, 11) normalised; text_vec (B, text_dim).
        With `return_aux`, also the auxiliary offset prediction (B, 8), in units of 3 m.
        With action tokens, motion is the greedy-decoded bin centre; `return_logits` also
        returns the (B, D, bins) action-token logits for the cross-entropy loss."""
        feats = self.fuse(*self.image_and_state(rgb_u8, proprio, text_vec), text_vec)
        motion, stop, logits = self.heads(feats)
        extra = []
        if return_aux:
            extra.append(self.aux_head(feats))
        if return_logits:
            extra.append(logits)
        return (motion, stop, *extra)

    def image_and_state(self, rgb_u8, proprio, text_vec=None):
        """(B, img_dim) image features and (B, 32) state features. The image path depends on
        the sentence only for the FiLM and attention variants."""
        x = rgb_u8.permute(0, 3, 1, 2).float().div(255.0).sub(0.5)
        fmap = self.cnn(x, text_vec)
        img = self.img_fc(self.attend(fmap, text_vec)) if self.attn else self.img_fc(fmap)
        return img, self.prop(proprio)

    @staticmethod
    def fuse(img, prop_feat, text_vec):
        return torch.cat([img, text_vec, prop_feat], dim=1)

    def heads(self, feats):
        """Fused features -> (motion (B, 4), Stop logit (B,), logits). For the token head the
        logits are (B, D, bins); for a chunk they are ((B, H, D, bins), (B, H) Stop logits)
        and motion/Stop are the first step's."""
        if self.chunk > 1:
            h = self.head_trunk(feats)                                         # (B, hid)
            out = self.head_out(torch.relu(h.unsqueeze(1) + self.step_emb))    # (B, H, n_out)
            logits = out[..., :-1].reshape(out.shape[0], self.chunk, len(self.TOKEN_DIMS),
                                           self.action_bins)
            stops = out[..., -1]
            return self.detokenize(logits[:, 0]), stops[:, 0], (logits, stops)
        out = self.head(feats)
        if self.action_bins:
            logits = out[:, :-1].reshape(out.shape[0], len(self.TOKEN_DIMS), self.action_bins)
            return self.detokenize(logits), out[:, -1], logits
        return torch.tanh(out[:, :4]), out[:, 4], None

    def decode_chunk(self, logits):
        """(B, H, D, bins) -> (B, H, 4) cap-normalised actions."""
        b, h = logits.shape[:2]
        return self.detokenize(logits.reshape(b * h, *logits.shape[2:])).reshape(b, h, 4)


def upgrade_state_dict(sd: dict) -> dict:
    """Checkpoints written before FiLM stored the conv stack as an nn.Sequential named `cnn`
    (convs at indices 0, 2, 4, 6, ReLUs between); it is now a ModuleList named `convs`."""
    out = {}
    for k, v in sd.items():
        if k.startswith("cnn."):
            idx, rest = k[len("cnn."):].split(".", 1)
            k = f"convs.{int(idx) // 2}.{rest}"
        out[k] = v
    return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def pad_batch(id_lists: list[list[int]]) -> torch.Tensor:
    n = max(len(x) for x in id_lists)
    return torch.tensor([x + [0] * (n - len(x)) for x in id_lists], dtype=torch.long)
