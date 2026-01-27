import random
import pandas as pd
from typing import Tuple, Optional

DNA = ("A", "C", "G", "T")
GAP = "-"


# -------------------------
# helpers
# -------------------------

def random_dna(length: int) -> str:
    return "".join(random.choices(DNA, k=length))


def insert_motif(seq: str, motif: str, pos: int) -> str:
    return seq[:pos] + motif + seq[pos + len(motif):]


def motif2_positions(seq: str, n: int):
    positions = []
    count = 0
    for i, ch in enumerate(seq):
        if ch in DNA:
            count += 1
            if count % n == 0:
                positions.append(i + 1)
    return positions


# -------------------------
# baseline generation
# -------------------------

def generate_baseline_sequence(
    base_len: int,
    flank_len: int,
    motif_A: str,
    motif_B: str,
    distance_unit: int
):
    core = random_dna(base_len)
    label = random.choice(["none", "A", "B", "both"])

    A_start = A_end = B_start = B_end = None

    if label == "A":
        pos = random.randint(0, base_len - len(motif_A))
        core = insert_motif(core, motif_A, pos)
        A_start, A_end = pos, pos + len(motif_A)

    elif label == "B":
        pos = random.randint(0, base_len - len(motif_B))
        core = insert_motif(core, motif_B, pos)
        B_start, B_end = pos, pos + len(motif_B)

    elif label == "both":
        max_start = base_len - (len(motif_A) + len(motif_B) + distance_unit)
        if max_start > 0:
            pos_A = random.randint(0, max_start)
            suffix = core[pos_A + len(motif_A):]
            offsets = motif2_positions(suffix, distance_unit)
            if offsets:
                pos_B = pos_A + len(motif_A) + random.choice(offsets)
                if pos_B <= base_len - len(motif_B):
                    core = insert_motif(core, motif_A, pos_A)
                    core = insert_motif(core, motif_B, pos_B)
                    A_start, A_end = pos_A, pos_A + len(motif_A)
                    B_start, B_end = pos_B, pos_B + len(motif_B)
                else:
                    label = "none"
            else:
                label = "none"
        else:
            label = "none"

    seq = random_dna(flank_len) + core + random_dna(flank_len)

    if A_start is not None:
        A_start += flank_len
        A_end += flank_len
    if B_start is not None:
        B_start += flank_len
        B_end += flank_len

    return seq, label, A_start, A_end, B_start, B_end


# -------------------------
# deletion augmentation
# -------------------------

def insert_deletions_and_track(seq: str, p: float):
    new_seq = []
    index_map = []  # maps old index -> new index (first occurrence)

    new_idx = 0
    for old_idx, ch in enumerate(seq):
        new_seq.append(ch)
        index_map.append(new_idx)
        new_idx += 1

        if ch in DNA:
            while random.random() < p:
                new_seq.append(GAP)
                new_idx += 1

    return new_seq, index_map


def trim_to_baseline_length(
    seq_list,
    target_len,
    A_start, A_end,
    B_start, B_end
):
    """
    Trim sequence by alternately removing first and last token
    until target_len is reached.
    Motif coordinates are shifted accordingly.
    """

    trim_left = True

    while len(seq_list) > target_len:
        if trim_left:
            # remove first token
            seq_list.pop(0)

            def shift(pos):
                if pos is None:
                    return None
                return pos - 1

            A_start = shift(A_start)
            A_end   = shift(A_end)
            B_start = shift(B_start)
            B_end   = shift(B_end)

        else:
            # remove last token
            seq_list.pop()
            # no coordinate shift needed

        trim_left = not trim_left

    return (
        "".join(seq_list),
        A_start, A_end,
        B_start, B_end
    )



def generate_deletion_sequence(row, deletion_prob):
    seq = row["sequences"]
    A_start = row["motif_A_start"]
    A_end   = row["motif_A_end"]
    B_start = row["motif_B_start"]
    B_end   = row["motif_B_end"]
    
    has_A = not pd.isna(A_start)
    has_B = not pd.isna(B_start)


    seq_list, index_map = insert_deletions_and_track(seq, deletion_prob)

    if has_A:
        A_start = index_map[int(A_start)]
        A_end = index_map[int(A_end) - 1] + 1
    else:
        A_start = A_end = None
    
    if has_B:
        B_start = index_map[int(B_start)]
        B_end = index_map[int(B_end) - 1] + 1
    else:
        B_start = B_end = None

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


# -------------------------
# main
# -------------------------

def generate_datasets(
    n_samples: int = 100_000,
    base_len: int = 100,
    flank_len: int = 20,
    motif_len_range: Tuple[int, int] = (5, 10),
    distance_unit: int = 10,
    deletion_prob: float = 0.15,
    out_prefix: str = "sequences"
):
    motif_A = random_dna(random.randint(*motif_len_range))
    motif_B = random_dna(random.randint(*motif_len_range))

    rows = []
    for _ in range(n_samples):
        seq, label, A_s, A_e, B_s, B_e = generate_baseline_sequence(
            base_len, flank_len, motif_A, motif_B, distance_unit
        )
        rows.append({
            "sequences": seq,
            "labels": label,
            "motif_A_start": A_s,
            "motif_A_end": A_e,
            "motif_B_start": B_s,
            "motif_B_end": B_e
        })

    df_baseline = pd.DataFrame(rows)
    df_baseline.to_parquet(f"baseline_{out_prefix}.parquet", index=False)

    df_del = df_baseline.apply(
        lambda r: generate_deletion_sequence(r, deletion_prob),
        axis=1
    )
    df_del.to_parquet(f"deletion_{out_prefix}.parquet", index=False)

    with open("motifs.txt", "w") as f:
        f.write(f"{motif_A}\n{motif_B}")


if __name__ == "__main__":
    generate_datasets()
