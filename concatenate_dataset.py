import os
from pathlib import Path

def concatenate_books(source_folder: Path, output_file: Path):
    files = []

    for p in sorted(source_folder.iterdir()):
        files.append(p.name)
    
    print("on concatène...")
    with open(output_file, 'w', encoding = 'utf-8') as output:
        for file_name in files:
            path = Path(f"{source_folder}/{file_name}")

            with open(path, 'r', encoding = 'utf-8') as file_text:
                output.write(file_text.read())
                output.write("\n\n")
    

    print("fini de tout coller")


if __name__ == "__main__":
    concatenate_books(source_folder = Path("victor"), output_file = "victor_complet.txt" )