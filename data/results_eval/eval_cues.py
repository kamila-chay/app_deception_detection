from pathlib import Path

source = Path("./cues")

total = 0
n = 0

for file in source.iterdir():
    with open(file, "r") as f:
        n += 1
        text = f.read()
        try:
            total += float(text)
        except:
            print(file.stem)

print(total / n)