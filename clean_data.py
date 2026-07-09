from pathlib import Path
import re


def clean_victor(input_path: Path, output_path: Path) -> None:
    assert input_path.exists(), "pas trouvé le fichier indiqué"

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Retirer les blocs de métadonnées de Project Gutenberg
    parts = re.split(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK", content, flags=re.IGNORECASE)
    content = parts[1] if len(parts) > 1 else parts[0]

    parts = re.split(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK", content, flags=re.IGNORECASE)
    content = parts[0]

    cleaned_lines = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        # Supprimer les lignes de boilerplate et les en-têtes typiques
        if re.match(
            r"^(title|author|release date|language|other information|credits|produced by|this ebook|this file was|end of project gutenberg|all rights reserved|collection|paris|librairie|rue|livre|pauca meæ|i|ii|iii|iv|v|vi|vii|viii|ix|x)\b",
            line,
            flags=re.IGNORECASE,
        ):
            continue

        line = re.sub(r"^\*+$", "", line)
        line = re.sub(r"\s+", " ", line)
        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)


if __name__ == "__main__":
    input_path = Path("victor/notre_dame_de_paris.txt")
    output_path = Path("victor/notre_dame_de_paris_clean.txt")
    clean_victor(input_path, output_path)
    print(f"Texte nettoyé écrit dans {output_path}")
