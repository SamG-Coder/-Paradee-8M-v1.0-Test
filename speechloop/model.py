"""Freshly initialized audio encoder. This is NOT Paradee's weights reversed."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .audio import N_MELS
from .text import ALPHABET


@dataclass
class ModelConfig:
    hidden: int = 128
    layers: int = 2
    conv_channels: int = 96
    dropout: float = 0.1
    # Optional auxiliary CTC head uses the SAME phoneme IDs as Paradee.
    # It shares the learned audio encoder, not the original TTS weights.
    phoneme_classes: int = 0


class TinyRecognizer(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        if min(c.hidden, c.layers, c.conv_channels) < 1:
            raise ValueError("Model dimensions must be positive")
        self.conv = nn.Sequential(nn.Conv1d(N_MELS, c.conv_channels, 5, stride=2, padding=2), nn.GELU())
        self.norm = nn.LayerNorm(c.conv_channels)
        self.rnn = nn.GRU(c.conv_channels, c.hidden, c.layers, batch_first=True,
                          bidirectional=True, dropout=c.dropout if c.layers > 1 else 0)
        self.head = nn.Linear(c.hidden * 2, len(ALPHABET))
        self.phoneme_head = nn.Linear(c.hidden * 2, c.phoneme_classes) if c.phoneme_classes else None

    @staticmethod
    def output_lengths(lengths: torch.Tensor) -> torch.Tensor:
        return (lengths + 1) // 2

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple:
        # Mask padding before AND after convolutions so batch padding cannot leak.
        lengths = lengths.cpu().long()
        mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths.to(x.device)[:, None]
        x = x * mask[:, :, None]
        x = self.norm(self.conv(x.transpose(1, 2)).transpose(1, 2))
        out_lengths = self.output_lengths(lengths)
        packed = pack_padded_sequence(x, out_lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.rnn(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=x.size(1))
        logits = self.head(encoded)
        phones = self.phoneme_head(encoded) if self.phoneme_head is not None else None
        return logits, out_lengths, phones

    def describe(self) -> dict:
        count = sum(p.numel() for p in self.parameters())
        return {"parameters": count, "fp32_parameter_bytes": count * 4, "config": asdict(self.config),
                "note": "Recognizer only; frozen TTS and runtime dependencies are additional."}
