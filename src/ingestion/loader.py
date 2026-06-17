import pandas as pd
import numpy as np
import os

def load_raw_data(data_dir: str) -> pd.DataFrame:
    """
    Finds all CSV files in data_dir, loads them, and combines into one DataFrame.
    """
    all_files = []

    # Loop through every file in the folder
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            print(f"Loading: {filename}")
            df = pd.read_csv(filepath, encoding="utf-8", low_memory=False)
            df.columns = df.columns.str.strip()
            all_files.append(df)

    # Combine all individual DataFrames into one
    combined = pd.concat(all_files, ignore_index=True)
    print(f"\nTotal rows loaded: {len(combined)}")
    print(f"Total columns: {len(combined.columns)}")
    return combined

def summarize_labels(df: pd.DataFrame) -> None:
    """
    Prints a count of each unique label in the dataset.
    This tells us what attack types exist and how balanced the data is.
    """
    print("\n--- Label Distribution ---")
    label_counts = df["Label"].value_counts()
    print(label_counts)
    print(f"\nTotal unique labels: {label_counts.nunique()}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw DataFrame:
    - Replaces infinite values with NaN, then drops those rows
    - Drops any remaining NaN rows
    - Adds a binary 'is_attack' column (0=benign, 1=attack)
    """
    print("\nCleaning data...")

    # Replace inf and -inf with NaN so we can drop them cleanly
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Count and drop rows with any NaN values
    before = len(df)
    df.dropna(inplace=True)
    after = len(df)
    print(f"Dropped {before - after} rows with NaN/inf values")

    # Create binary label: 0 for BENIGN, 1 for any attack
    df["is_attack"] = (df["Label"] != "BENIGN").astype(int)
    print(f"Attack rows: {df['is_attack'].sum()}")
    print(f"Benign rows: {(df['is_attack'] == 0).sum()}")

    return df

def save_processed(df: pd.DataFrame, output_path: str) -> None:
    """
    Saves the cleaned DataFrame to a CSV file in data/processed/
    """
    df.to_csv(output_path, index=False)
    print(f"\nSaved cleaned data to: {output_path}")

if __name__ == "__main__":
    data_dir = "data/raw"
    df = load_raw_data(data_dir)
    print(df.columns.tolist())
    summarize_labels(df)
    df = clean_data(df)
    save_processed(df, "data/processed/cleaned.csv")