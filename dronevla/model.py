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
    def __init__(self, vocab_size: int, text_dim: int = 32, proprio_dim: int = 11):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),     # 48 x 64
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),    # 24 x 32
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),    # 12 x 16
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),    #  6 x  8
        )
        self.img_fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 6 * 8, 128), nn.ReLU())
        self.embed = nn.Embedding(vocab_size, text_dim, padding_idx=0)
        self.prop = nn.Sequential(nn.Linear(proprio_dim, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128 + text_dim + 32, 128), nn.ReLU(),
                                  nn.Linear(128, 5))

    def encode_text(self, ids: torch.Tensor) -> torch.Tensor:
        """(B, T) token ids, 0 = pad -> (B, text_dim) mean over real tokens."""
        mask = (ids != 0).unsqueeze(-1).float()
        return (self.embed(ids) * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def forward(self, rgb_u8: torch.Tensor, proprio: torch.Tensor, text_vec: torch.Tensor):
        """rgb_u8 (B, 96, 128, 3) uint8; proprio (B, 11) normalised; text_vec (B, text_dim)."""
        x = rgb_u8.permute(0, 3, 1, 2).float().div(255.0).sub(0.5)
        feats = torch.cat([self.img_fc(self.cnn(x)), text_vec, self.prop(proprio)], dim=1)
        out = self.head(feats)
        return torch.tanh(out[:, :4]), out[:, 4]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def pad_batch(id_lists: list[list[int]]) -> torch.Tensor:
    n = max(len(x) for x in id_lists)
    return torch.tensor([x + [0] * (n - len(x)) for x in id_lists], dtype=torch.long)
