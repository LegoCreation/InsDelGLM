import pandas as pd
import re
import random
 
# change this to the file locations of your data which need adaption
data_files = ["train.parquet", "val.parquet", "test.parquet"]
# known motifs
motifs = ["CGTAGGTC", "GTAACGCTCC"]

pattern = re.compile(
    rf"{motifs[0]}((?:(?:-*[ACGT]-*){{10}})+){motifs[1]}"
)


def adapt_dataset():
    for data_file in data_files:
        df = pd.read_parquet(data_file)        
        df_save = df.apply(handle_deletions, axis=1)
        df_save.to_parquet(f"baseline_{data_file}", index=False)

def handle_deletions(row):
    seq = list(row["sequences"])
    if not row["labels"] == "both":
        seq = substitute_all_deletions(seq)
    else:
        match = pattern.search(row["sequences"])
        if match is None:
            print(row["id"])
            return row  # skip rows without a match
        start, end = match.span()

        matched_seq = row["sequences"][start:end]
        num_bases = sum(c in "ACGT" for c in matched_seq)
        
        for i in range(end - start - num_bases):
            seq[end-1-i] = "-"

        for i, n in enumerate(motifs[1]):
            seq[start - len(motifs[1]) + num_bases + i] = n

        seq = substitute_all_deletions(seq)

    row["sequences"] = seq
    return row


def substitute_all_deletions(sequence):
    bases = ("A", "C", "T", "G")
    return "".join(
        random.choice(bases) if c == "-" else c
        for c in sequence
    )


if __name__ == "__main__":
    adapt_dataset()