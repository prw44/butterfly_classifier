import pandas as pd

df = pd.read_csv("data/Training_set.csv")
classes = sorted(df.iloc[:, 1].unique())

with open("models/classes.txt", "w") as file:
    file.write("\n".join(classes))


from dataset import CustomDataset

dataset = CustomDataset(csv_file="data/Training_set.csv", img_dir="data/train")

assert classes == dataset.classes

