import pandas as pd
from sklearn.model_selection import train_test_split
import os


def split_dataset(
        input_file="./simulated/deletion_sequences.parquet",
        output_dir="./simulated",
        seed=42
):
    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_file)

    # 1. Add ID (simple row number) to keep track of original indices
    df["id"] = range(len(df))

    print(f"Total number of sequences: {len(df)}")
    print("Label distribution (Total):")
    print(df["labels"].value_counts(normalize=True))

    # 2. Split: First remove 20% for the Test set
    # 'stratify' ensures that the label distribution (A, B, both) stays consistent across splits
    df_temp, df_test = train_test_split(
        df,
        test_size=0.2,
        stratify=df["labels"],
        random_state=seed
    )

    # 3. Split: From the remaining 80%, take 25% for Validation
    # (Since 80% * 0.25 = 20% of the total, this results in a 60/20/20 split)
    df_train, df_val = train_test_split(
        df_temp,
        test_size=0.25,
        stratify=df_temp["labels"],
        random_state=seed
    )

    # Verify the split sizes
    n_total = len(df)
    print(f"\n--- Split Results ---")
    print(f"Train: {len(df_train)} ({len(df_train) / n_total:.1%})")
    print(f"Val:   {len(df_val)} ({len(df_val) / n_total:.1%})")
    print(f"Test:  {len(df_test)} ({len(df_test) / n_total:.1%})")

    # Save the files
    print(f"\nSaving files to {output_dir}...")
    df_train.to_parquet(os.path.join(output_dir, "deletion_train.parquet"))
    df_val.to_parquet(os.path.join(output_dir, "deletion_val.parquet"))
    df_test.to_parquet(os.path.join(output_dir, "deletion_test.parquet"))

    print("Done!")


if __name__ == "__main__":
    split_dataset()