from pathlib import Path

source = Path("./final_label")

total = 0
n = 0

for file in source.iterdir():
    with open(file, "r") as f:
        n += 1
        text = f.read()
        try:
            total += float(text.split("Score:")[1])

        except (ValueError, IndexError):
            try:
                total += float(text.split("\n")[-1])
            except (ValueError, IndexError):
                print(file.stem)

print(total / n)
