import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.optim import AdamW
from tokenizers import Tokenizer
import json

from language_model_from_scratch import language_model
from dataset import Hugo_PretokenizedDataset

#config model
num_layers = 3
d_hidden= 1024
n_heads = 32
n_groups = 8

#config training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 5
batch_size = 16
seq_len = 512
max_grad_norm = 1.0

learning_rate = 6e-4
weight_decay = 0.1
beta1, beta2 = 0.9, 0.95

tokenizer = Tokenizer.from_file("hugo_tokenizer.json")

with open("victor_complet.txt", "r", encoding="utf-8") as f:
    text = f.read()

# split 90/10 (par caractères). Si tu préfères par lignes, utilise text.splitlines()
split_idx = int(len(text) * 0.9)
train_text = text[:split_idx]
eval_text = text[split_idx:]

train_ids = tokenizer.encode(train_text).ids
eval_ids = tokenizer.encode(eval_text).ids

train_ds = Hugo_PretokenizedDataset(train_ids, seq_len=seq_len, trainer=True)
eval_ds = Hugo_PretokenizedDataset(eval_ids, seq_len=seq_len, trainer=False)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)
eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2)

# 3. Instanciation du Modèle
model = language_model(num_layers = num_layers, n_heads = n_heads, n_groups = n_groups, d_hidden = d_hidden, vocab_size = tokenizer.get_vocab_size()).to(device)

# 4. Groupement des Paramètres (Weight Decay Sélectif)
# Séparation stricte : Matrices 2D (Decay) vs Vecteurs 1D / Biais / RMSNorm scale (No Decay)
param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.ndim >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.ndim < 2]

optim_groups = [
    {"params": decay_params, "weight_decay": weight_decay},
    {"params": nodecay_params, "weight_decay": 0.0}
]

optimizer = AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2), eps=1e-8)

# 5. Boucle Principale
for epoch in range(num_epochs):
    train_ds.training_shuffler()
    model.train()
    total_train_loss = 0.0
    
    for x, y in train_loader:
        x = x.long().to(device, non_blocking=True)
        y = y.long().to(device, non_blocking=True)

        #print(f"processing {x} and {y}")

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)  # [B, T, vocab_size]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        print(f"loss = {loss}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        total_train_loss += loss.item()

    # Phase de Validation
    model.eval()
    eval_ds.training_shuffler()  
    total_val_loss = 0.0
    
    with torch.no_grad():
        for x, y in eval_loader:
            x = x.long().to(device, non_blocking=True)
            y = y.long().to(device, non_blocking=True)
            
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                
            total_val_loss += val_loss.item()
            
    print(f"Epoch {epoch+1} | Train Loss: {total_train_loss/len(train_loader):.4f} | Val Loss: {total_val_loss/len(eval_loader):.4f}")