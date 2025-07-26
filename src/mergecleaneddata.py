import pandas as pd
import glob
import os

# Folder where your cleaned CSV files are
csv_folder = './data/cleaned/'
csv_files = glob.glob(csv_folder + "*.csv")

merged_df = None

for file in csv_files:
    df = pd.read_csv(file)
    df['Area'] = df['Area'].astype(str).str.strip()

    if merged_df is None:
        merged_df = df
    else:
        # Merge while dropping duplicate columns from the new DataFrame
        df = df.drop(columns=[col for col in df.columns if col in merged_df.columns and col != "Area"])
        merged_df = pd.merge(merged_df, df, on='Area', how='outer')

# Fill missing values if needed
merged_df.fillna(0, inplace=True)

# Save
merged_df.to_csv("merged_cleaned_dataset.csv", index=False)
