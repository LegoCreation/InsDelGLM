import pandas as pd
import re

# ------------------
# config
# ------------------

BASELINE_FILE = "baseline_sequences.parquet"
DELETION_FILE = "deletion_sequences.parquet"
MOTIF_FILE = "motifs.txt"
DISTANCE_UNIT = 6 #10

DNA = set("ACGT")
DNA_GAP = set("ACGT-")


# ------------------
# helpers
# ------------------

def biological_distance(seq, end_A, start_B):
    return sum(
        c in DNA
        for c in seq[end_A:start_B]
    )


def motif_matches_with_gaps(seq, motif):
    """
    Checks if motif appears in seq allowing gaps between characters
    """
    pattern = ".*?".join(motif)
    return re.search(pattern, seq) is not None


# ------------------
# load data
# ------------------

df_base = pd.read_parquet(BASELINE_FILE)
df_del = pd.read_parquet(DELETION_FILE)

with open(MOTIF_FILE) as f:
    motif_A, motif_B = [line.strip() for line in f]


# ------------------
# basic checks
# ------------------

assert len(df_base) == len(df_del), "Row count mismatch"
assert list(df_base.columns) == list(df_del.columns), "Column mismatch"

seq_len = df_base["sequences"].str.len().unique()
assert len(seq_len) == 1, "Baseline sequences have varying lengths"

assert (df_del["sequences"].str.len() == seq_len[0]).all(), \
    "Deletion sequences length mismatch"


# ------------------
# per-row checks (sampled)
# ------------------

sample = df_base.sample(min(500, len(df_base)), random_state=1).index

for idx in sample:
    row_b = df_base.loc[idx]
    row_d = df_del.loc[idx]

    seq_b = row_b.sequences
    seq_d = row_d.sequences

    # alphabet
    assert set(seq_b).issubset(DNA), f"Invalid char in baseline at {idx}"
    assert set(seq_d).issubset(DNA_GAP), f"Invalid char in deletion at {idx}"

    for motif, start, end, name in [
        (motif_A, row_b.motif_A_start, row_b.motif_A_end, "A"),
        (motif_B, row_b.motif_B_start, row_b.motif_B_end, "B"),
    ]:
        if pd.isna(start):
            assert pd.isna(end), f"Half-missing motif {name} at {idx}"
            continue

        start, end = int(start), int(end)
        assert 0 <= start < end <= len(seq_b), f"Bad coords {name} at {idx}"

        # baseline exact match
        assert seq_b[start:end] == motif, \
            f"Baseline motif {name} mismatch at {idx}"

        # deletion motif still present (with gaps)
        assert motif_matches_with_gaps(seq_d, motif), \
            f"Motif {name} lost in deletion at {idx}"

    # biological distance check
    if row_b.labels == "both":
        d = biological_distance(
            seq_d,
            int(row_d.motif_A_end),
            int(row_d.motif_B_start)
        )
        assert d % DISTANCE_UNIT == 0, \
            f"Distance violation at {idx}: {d}"


print("Yupp, All checks passed.")
