from pathlib import Path

import numpy as np
from scipy import stats

source = Path("./cues")

all_scores = []

for file in source.iterdir():
    with open(file, "r") as f:
        text = f.read()
        try:
            all_scores.append(float(text))
        except ValueError:
            print(file.stem)

data = np.array(all_scores)

mu0 = 0.8

t_stat, p_val = stats.ttest_1samp(data, mu0)
print(f"t = {t_stat:.3f}, p = {p_val:.4f}")
