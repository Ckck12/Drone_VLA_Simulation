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
    It exists so the BC-vs-world-model comparison gives both the same privileged supervision."""

    CHANNELS = (16, 32, 64, 64)

    def __init__(self, vocab_size: int, text_dim: int = 32, proprio_dim: int = 11,
                 film: bool = False, aux_offsets: bool = False):
        super().__init__()
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
        self.img_fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 6 * 8, 128), nn.ReLU())
        self.embed = nn.Embedding(vocab_size, text_dim, padding_idx=0)
        self.prop = nn.Sequential(nn.Linear(proprio_dim, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128 + text_dim + 32, 128), nn.ReLU(),
                                  nn.Linear(128, 5))
        self.aux_head = (nn.Sequential(nn.Linear(128 + text_dim + 32, 64), nn.ReLU(),
                                       nn.Linear(64, 8)) if aux_offsets else None)

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

    def encode_text(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, T) token ids, 0 = pad -> (B, text_dim) mean over real tokens."""
        mask = (ids != 0).unsqueeze(-1).float()
        return (self.embed(ids) * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def forward(self, rgb_u8: torch.Tensor, proprio: torch.Tensor, text_vec: torch.Tensor,
                return_aux: bool = False):
        """rgb_u8 (B, 96, 128, 3) uint8; proprio (B, 11) normalised; text_vec (B, text_dim).
        With `return_aux`, also the auxiliary offset prediction (B, 8), in units of 3 m."""
        x = rgb_u8.permute(0, 3, 1, 2).float().div(255.0).sub(0.5)
        feats = torch.cat([self.img_fc(self.cnn(x, text_vec)), text_vec, self.prop(proprio)],
                          dim=1)
        out = self.head(feats)
        if return_aux:
            return torch.tanh(out[:, :4]), out[:, 4], self.aux_head(feats)
        return torch.tanh(out[:, :4]), out[:, 4]


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
