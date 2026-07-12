import torch
from torch.utils.data import Dataset, DataLoader
import json

class Hugo_PretokenizedDataset(Dataset):
    def __init__(self, tokens, seq_len: int = 512, trainer=True):
        super().__init__()
        # tokens peut être une list[int] ou un tenseur ; on normalise en LongTensor
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = int(seq_len)
        self.trainer = bool(trainer)
        self.offset = 0
        self.num_samples = max(0, (len(self.tokens) - 1) // self.seq_len)

    def training_shuffler(self):
        if self.trainer and len(self.tokens) > 1:
            # offset dans [0, seq_len-1], mais pas plus long que len(tokens)-2
            max_offset = min(self.seq_len - 1, max(0, len(self.tokens) - 2))
            self.offset = torch.randint(0, max_offset + 1, (1,)).item() if max_offset > 0 else 0
            self.num_samples = max(0, (len(self.tokens) - 1 - self.offset) // self.seq_len)

    def __len__(self):
        return max(0, self.num_samples)

    def __getitem__(self, idx):
        if idx < 0:
            idx = self.num_samples + idx
        if idx < 0 or idx >= self.num_samples:
            raise IndexError("Index out of range")
        start_idx = self.offset + (idx * self.seq_len)
        end_idx = start_idx + self.seq_len

        # slicing sur LongTensor -> tensors longs
        x = self.tokens[start_idx:end_idx]
        y = self.tokens[start_idx + 1:end_idx + 1]

        # garantir la bonne taille (peut être utile pour debug)
        if x.shape[0] != self.seq_len or y.shape[0] != self.seq_len:
            raise RuntimeError(f"Unexpected sample length: got {x.shape[0]} (expected {self.seq_len})")

        return x, y