import pandas as pd

main = pd.read_excel("data/traits.xlsx")
extra = pd.read_excel("/home/kamila/work/publication1/workspace/data_meta/dolos/Dolos.xlsx")

merged = main.merge(extra[["Filename", "Participants name"]], on="Filename", how="left")

merged.to_excel("data/traits_merged.xlsx", index=False)

print(merged.isna().any().any())