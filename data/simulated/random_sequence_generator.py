import random
import pandas as pd

DNA = ["A", "C", "G", "T"]


def random_dna(length):
    return "".join(random.choices(DNA, k=length))


def insert_motif(seq, motif, pos):
    return seq[:pos] + motif + seq[pos + len(motif):]


def add_gap_token(seq, gap_prob=0.05):
    seq = list(seq)
    for i in range(len(seq)):
        if random.random() < gap_prob:
            seq[i] = "-"
    return "".join(seq)

def motif2_positions(seq, n=10):
    positions = []
    count = 0

    for i, ch in enumerate(seq):
        if ch in DNA:
            count += 1
            if count % n == 0:
                positions.append(i+1)

    return positions


def generate_dataset(
    n_samples=5000,
    seq_len=512,
    motif_len_range=(5, 10),
    gap_prob=0.05
):
    # Generate motifs
    motif_A = random_dna(random.randint(*motif_len_range))
    motif_B = random_dna(random.randint(*motif_len_range))

    labels = []
    sequences = []

    for _ in range(n_samples):
        seq = random_dna(seq_len)
        seq = add_gap_token(seq, gap_prob)

        place_both = random.choice([True, False])

        if place_both:
            # Valid spacing: multiple of 10
            max_start = seq_len - (len(motif_A) + len(motif_B) + 10)
            pos_A = random.randint(0, max_start)

            # possible motif B positions from pos_A on
            m2_positions = motif2_positions(seq=seq[pos_A+len(motif_A):])
            if m2_positions:
                pos_B = pos_A + len(motif_A) + random.choice(m2_positions)
            else:
                pos_B = float('inf')

            if pos_B <= seq_len - len(motif_B):
                seq = insert_motif(seq, motif_A, pos_A)
                seq = insert_motif(seq, motif_B, pos_B)
                labels.append("both")  # both motifs
            else:
                place_both = False  # fallback

        if not place_both:
            # Insert only one motif
            motif = random.choice([motif_A, motif_B])
            pos = random.randint(0, seq_len - len(motif))
            seq = insert_motif(seq, motif, pos)
            if motif == motif_A:
                labels.append("A")
            elif motif == motif_B:
                labels.append("B")
            else:
                labels.append("ERROR")



        sequences.append(seq)

    df = pd.DataFrame({
        'sequences': sequences,
        'labels': labels  # only for interpretation, not for training
    })

    df.to_parquet("./simulated_sequences.parquet", index=False)

    with open("motifs.txt", "w") as f:
        f.write(f"{motif_A}\n{motif_B}")


if __name__ == "__main__":
    generate_dataset()
