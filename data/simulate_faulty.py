import pandas as pd
import random
import os

from data.simulated.biological_seqgen import (
    insert_deletions_and_track,
    trim_to_baseline_length,
    random_dna,
    insert_motif
)

# --- Configuration ---
NUM_SEQUENCES = 10000
SEQ_LENGTH = 140
DELETION_PROB = 0.15
GAP = "-"
DNA = ("A", "C", "G", "T")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOTIF_PATH = os.path.join(BASE_DIR, 'simulated/motifs.txt')
OUTPUT_BASELINE = os.path.join(BASE_DIR, 'faulty_baseline.parquet')
OUTPUT_DELETION = os.path.join(BASE_DIR, 'faulty_deletions.parquet')


# --- Logic ---

def load_motifs(path):
    """Loads Motif A and B from the specified file."""
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        if len(lines) < 2:
            raise ValueError("File must contain at least two lines for Motif A and B")
        return lines[0], lines[1]


def generate_faulty_baseline_sequence(seq_len, motif_a, motif_b):
    """
    Generate a single faulty sequence where motifs are NOT at biological distances (10*N).
    Returns: seq, A_start, A_end, B_start, B_end
    """
    core = random_dna(seq_len)

    len_a = len(motif_a)
    len_b = len(motif_b)

    # Calculate maximum spacer length
    max_spacer = seq_len - len_a - len_b
    if max_spacer < 0:
        raise ValueError(f"Motifs too long for sequence length {seq_len}")

    # Faulty logic: Distances NOT multiples of 10
    faulty_spacers = [d for d in range(0, max_spacer + 1) if d % 10 != 0]

    if not faulty_spacers:
        raise ValueError("No valid faulty spacers available with current parameters")

    # Choose a random faulty spacer
    spacer_len = random.choice(faulty_spacers)

    # Calculate block length and random starting position
    block_len = len_a + spacer_len + len_b
    pos_A = random.randint(0, seq_len - block_len)

    # Insert motif A
    core = insert_motif(core, motif_a, pos_A)
    A_start = pos_A
    A_end = pos_A + len_a

    # Insert motif B at the faulty distance
    pos_B = pos_A + len_a + spacer_len
    core = insert_motif(core, motif_b, pos_B)
    B_start = pos_B
    B_end = pos_B + len_b

    return core, A_start, A_end, B_start, B_end


def generate_deletion_sequence(row, deletion_prob):
    """
    Apply deletion augmentation to a baseline sequence.
    Reuses the logic from biological_seqgen.
    """
    seq = row["sequences"]
    A_start = row["motif_A_start"]
    A_end = row["motif_A_end"]
    B_start = row["motif_B_start"]
    B_end = row["motif_B_end"]

    # Insert deletions and track index mapping
    seq_list, index_map = insert_deletions_and_track(seq, deletion_prob)

    # Map coordinates to new positions
    A_start = index_map[int(A_start)]
    A_end = index_map[int(A_end) - 1] + 1
    B_start = index_map[int(B_start)]
    B_end = index_map[int(B_end) - 1] + 1

    # Trim to original length
    final_seq, A_start, A_end, B_start, B_end = trim_to_baseline_length(
        seq_list,
        len(seq),
        A_start, A_end,
        B_start, B_end
    )

    row["sequences"] = final_seq
    row["motif_A_start"] = A_start
    row["motif_A_end"] = A_end
    row["motif_B_start"] = B_start
    row["motif_B_end"] = B_end
    return row


def generate_faulty_dataset(num_seqs, seq_len, motif_a, motif_b):
    """
    Generate baseline and deletion datasets for faulty sequences.
    """
    rows = []

    for i in range(num_seqs):
        seq, A_s, A_e, B_s, B_e = generate_faulty_baseline_sequence(
            seq_len, motif_a, motif_b
        )
        rows.append({
            'id': i,
            'sequences': seq,
            'labels': 'faulty',
            'motif_A_start': A_s,
            'motif_A_end': A_e,
            'motif_B_start': B_s,
            'motif_B_end': B_e
        })

    df_baseline = pd.DataFrame(rows)

    # Apply deletion augmentation
    df_deletion = df_baseline.copy()
    df_deletion = df_deletion.apply(
        lambda r: generate_deletion_sequence(r, DELETION_PROB),
        axis=1
    )

    return df_baseline, df_deletion


if __name__ == "__main__":
    try:
        # 1. Load Motifs
        if not os.path.exists(MOTIF_PATH):
            print(f"Warning: {MOTIF_PATH} not found. Please ensure motifs.txt exists.")
        else:
            m_a, m_b = load_motifs(MOTIF_PATH)
            print(f"Reading motifs from: {MOTIF_PATH}")
            print(f"Motif A: {m_a} | Motif B: {m_b}")

            # 2. Generate Data
            print(f"Generating {NUM_SEQUENCES} faulty sequences (Baseline vs Deletion versions)...")
            df_base, df_del = generate_faulty_dataset(NUM_SEQUENCES, SEQ_LENGTH, m_a, m_b)

            # --- FILTERING LOGIC ---
            print(f"Original Deletion Dataset Size: {len(df_del)}")

            # Keep rows where:
            # 1. Motif A hasn't been trimmed from the left (start >= 0)
            # 2. Motif B hasn't been trimmed from the right (end <= SEQ_LENGTH)
            df_del_filtered = df_del[
                (df_del['motif_A_start'] >= 0) &
                (df_del['motif_B_end'] <= SEQ_LENGTH)
                ].copy()

            print(f"Filtered Deletion Dataset Size: {len(df_del_filtered)}")

            # 3. Save to Parquet
            df_base.to_parquet(OUTPUT_BASELINE, index=False)
            df_del_filtered.to_parquet(OUTPUT_DELETION, index=False)

            print("\nDone!")
            print(f"Baseline file (Raw DNA): {OUTPUT_BASELINE}")
            print(f"Deletion file (Valid only): {OUTPUT_DELETION}")

    except Exception as e:
        print(f"Error during execution: {e}")