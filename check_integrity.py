from pathlib import Path

import pandas as pd

data_folder = Path("./data/video")
traits_file = Path("./data/traits.xlsx")
traits_file = pd.read_excel(traits_file)
print(traits_file.columns.tolist())

vidoes_data_folder = []
videos_traits_file = set()

for file in data_folder.iterdir():
    vidoes_data_folder.append(file.stem.strip())

for i, row in traits_file.iterrows():
    filename = (row.loc["File name of the video clip "]).strip()
    if filename in videos_traits_file:
        print(f"Duplicate: {filename}, {i}")
    else:
        videos_traits_file.add(filename)

if len(vidoes_data_folder) != len(set(vidoes_data_folder)):
    print("Duplicates in the data folder")

if len(videos_traits_file) != len(set(videos_traits_file)):
    print("Duplicates in the traits file")


vidoes_data_folder = set(vidoes_data_folder)

# print(len(vidoes_data_folder))
# print(len(videos_traits_file))


for video in vidoes_data_folder:
    if video not in videos_traits_file:
        print(video)  # nothing here

invalid_videos_in_traits = list()

for video in videos_traits_file:
    if video not in vidoes_data_folder:
        invalid_videos_in_traits.append(video)

print("Still invalid")
print(invalid_videos_in_traits)

# print("=======================")

# print(traits_file['File name of the video clip '])

# filtered_traits = traits_file[~traits_file['File name of the video clip '].isin(invalid_videos_in_traits)] # somthing goes wrong here
# filtered_traits.to_excel("./data/traits_filtered.xlsx", index=False)
