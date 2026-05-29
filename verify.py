import pandas as pd

df = pd.read_csv("data/raw/clinical.csv")

print(f"Dataset loaded: {len(df)} records")
print(f"Classes: {df['dr_grade'].value_counts().to_dict()}")
print(f"Splits: {df['split'].value_counts().to_dict()}")
print("Data verification complete!")