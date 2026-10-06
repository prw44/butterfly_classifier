import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/Training_set.csv")

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df.iloc[:, 1]
)

train_df.to_csv("data/Training_split.csv", index=False)
val_df.to_csv("data/Validation_split.csv", index=False)

print(f"Training: {len(train_df)}")
print(f"Validation: {len(val_df)}")