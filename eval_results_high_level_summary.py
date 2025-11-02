from pathlib import Path

root = Path("data/high_level_eval")
scores_labels = []
scores_cues = []

for file in root.iterdir():
    with open(file, "r") as f:
        try:
            score_label, score_cues = map(float, f.read().split())
            scores_labels.append(score_label)
            scores_cues.append(score_cues)
        except:
            print(f"Error with {file.stem}")

print("For labels:")
print(sum(scores_labels) / len(scores_labels))
print("For cues:")
print(sum(scores_cues) / len(scores_cues))

