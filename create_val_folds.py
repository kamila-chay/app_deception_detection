import pandas as pd

for i in range(1, 4):
    df = pd.read_csv(f"data/train_fold{i}.csv")

    every_15th = df.iloc[14::15]  
    rest = df.drop(every_15th.index)

    every_15th.to_csv(f"data/val_fold{i}.csv", index=False)
    rest.to_csv(f"data/train_fold{i}.csv", index=False)