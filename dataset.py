import torch
from torch.utils.data import Dataset, DataLoader
import json

class Hugo_PretokenizedDataset(Dataset):
    def __init__(self, text_file_path, tokenizer, seq_len: int = 512, trainer = True):
        super().__init__()

        self.seq_len = seq_len
        self.trainer = trainer

        with open(text_file_path, 'r', encoding = 'utf-8') as f:
            raw_text = f.read()
        
        tokenized = tokenizer.encode(raw_text)

        #liste des tokens de notre fichier
        self.tokens = torch.tensor(encoded.ids, dtype = torch.long)

        self.num_samples = (len(self.tokens)-1)// self.seq_len
        self.offset = 0
    
    def training_shuffler(self):
        if self.trainer:
            self.offset = torch.randint(0, self.seq_len, (1,)).item()
            self.num_samples = (len(self.tokens) - 1 - self.offset) // self.seq_len
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # calcul de la position de départ dans le grand vecteur de tokens
        start_idx = self.offset + (idx * self.seq_len)
        end_idx = start_idx + self.seq_len
        
        # extraction de X et Y (décalé de 1)
        x = self.tokens[start_idx:end_idx]
        y = self.tokens[start_idx+1:end_idx+1]
        
        return x, y