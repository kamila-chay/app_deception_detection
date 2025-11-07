from pathlib import Path

root = Path("out/perception_lm_v1_test_sets_high_level_eval")
scores_labels = []
scores_cues = []

false_pos = 0
false_neg = 0
true_pos = 0
true_neg = 0
for file in root.iterdir():
    with open(file, "r") as f:
        try:
            score_label, score_cues = map(float, f.read().split())
            scores_labels.append(score_label)
            scores_cues.append(score_cues)
            if score_label == 0:
                if "lie" in file.stem or "deception" in file.stem:
                    false_neg += 1
                else:
                    false_pos += 1
            else:
                if "lie" in file.stem or "deception" in file.stem:
                    true_pos += 1
                else:
                    true_neg += 1
        except ValueError:
            print(f"Error with {file.stem}")

print("Accuracy for labels:")
print(sum(scores_labels) / len(scores_labels))
print("Quality of cues:")
print(sum(scores_cues) / len(scores_cues))

f1 = (2 * true_pos) / (2 * true_pos + false_pos + false_neg)

print("F1 for labels:")
print(f1)
