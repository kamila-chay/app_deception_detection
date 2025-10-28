from pathlib import Path
import pandas as pd

existing_files = pd.read_excel("data/traits.xlsx")["Filename"].tolist()
if len(existing_files) != len(set(existing_files)):
    print("Error")

existing_files = set(existing_files)

for fold_file in {"train_fold1.csv", "train_fold2.csv", "train_fold3.csv", "test_fold1.csv", "test_fold2.csv", "test_fold3.csv"}:
    df = pd.read_csv(f"data/{fold_file}")
    files = df.iloc[:, 0].tolist()
    sett = set()
    print(fold_file)
    for file in files:
        if file in sett:
            print(file)
        else:
            sett.add(file)
        if file not in existing_files:
            print(file)
    