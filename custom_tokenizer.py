from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

dossier = Path("victor")
fichiers = []

for p in sorted(dossier.iterdir()):
    if p.is_file():
        print(f"ajout de {p.name} aux fichiers")
        fichiers.append(f"./victor/" +p.name)

print("init du tokenizer \n")
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

print("training...")
trainer =  BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"], vocab_size= 8192)
tokenizer.train(files = fichiers, trainer=trainer)

print("saving...")
tokenizer.save("hugo_tokenizer.json")

print("tokenizer saved hehe :3")