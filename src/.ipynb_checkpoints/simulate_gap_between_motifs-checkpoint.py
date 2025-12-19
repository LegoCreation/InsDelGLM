import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

DNA = ["A", "C", "G", "T"]

def random_dna(n: int) -> str:
    return "".join(random.choice(DNA) for _ in range(n))

def overwrite(seq: List[str], start: int, s: str) -> None:
    """Overwrite seq[start:start+len(s)] with string s (keeps length fixed)."""
    for i, ch in enumerate(s):
        seq[start + i] = ch

@dataclass
class SimConfig:
    length: int = 200
    motif_a: str = "ATATTCA"
    motif_b: str = "GTACTGC"
    spacing_multiple: int = 10
    # choose spacing d from this range of multiples: d = k*spacing_multiple
    k_range: Tuple[int, int] = (1, 8)  # k=1..8 => d=10..80 for spacing_multiple=10
    # dataset composition (analysis labels)
    p_both_valid: float = 0.5
    p_a_only: float = 0.25
    p_b_only: float = 0.25
    # gap insertion control (ONLY between motifs)
    gap_rate_between: float = 0.15   # probability per position in the between region to turn into '-'
    gap_min: int = 0                 # optional hard min number of gaps (0 means no min)
    gap_max: int = 999999            # optional hard max number of gaps
    seed: int = 1

def sample_case(cfg: SimConfig) -> str:
    r = random.random()
    if r < cfg.p_both_valid:
        return "both_valid"
    elif r < cfg.p_both_valid + cfg.p_a_only:
        return "A_only"
    else:
        return "B_only"

def sample_positions_both(cfg: SimConfig) -> Dict[str, int]:
    """Sample (a_start, b_start) that ALWAYS fit and satisfy valid spacing."""
    a_len, b_len = len(cfg.motif_a), len(cfg.motif_b)
    k = random.randint(cfg.k_range[0], cfg.k_range[1])
    d = k * cfg.spacing_multiple  # spacing between motifs

    # We define spacing as: b_start = a_start + a_len + d
    # Ensure it fits: a_start + a_len + d + b_len <= length
    max_a = cfg.length - (a_len + d + b_len)
    if max_a < 0:
        raise ValueError("Config impossible: sequence too short for motifs + spacing.")
    a_start = random.randint(0, max_a)
    b_start = a_start + a_len + d
    return {"a_start": a_start, "b_start": b_start, "spacing": d}

def sample_position_single(cfg: SimConfig, which: str) -> int:
    motif = cfg.motif_a if which == "A" else cfg.motif_b
    max_start = cfg.length - len(motif)
    return random.randint(0, max_start)

def add_gaps_between(seq: List[str], start: int, end: int, cfg: SimConfig) -> int:
    """
    Replace some characters with '-' ONLY in [start, end) region.
    Returns number of gaps inserted.
    """
    if end <= start:
        return 0
    idxs = []
    for i in range(start, end):
        if random.random() < cfg.gap_rate_between:
            idxs.append(i)

    # apply min/max constraints if desired
    if cfg.gap_min > 0 and len(idxs) < cfg.gap_min:
        # add extra random positions
        candidates = list(range(start, end))
        random.shuffle(candidates)
        for i in candidates:
            if i not in idxs:
                idxs.append(i)
            if len(idxs) >= cfg.gap_min:
                break

    if len(idxs) > cfg.gap_max:
        idxs = random.sample(idxs, k=cfg.gap_max)

    for i in idxs:
        seq[i] = "-"
    return len(idxs)

def generate_one(cfg: SimConfig) -> Dict:
    seq = list(random_dna(cfg.length))
    case = sample_case(cfg)

    record = {
        "case": case,
        "a_start": -1,
        "b_start": -1,
        "spacing": None,
        "gap_count": 0,
        "gap_region": None,
    }

    if case == "both_valid":
        pos = sample_positions_both(cfg)
        a_start, b_start, spacing = pos["a_start"], pos["b_start"], pos["spacing"]

        overwrite(seq, a_start, cfg.motif_a)
        overwrite(seq, b_start, cfg.motif_b)

        # gap ONLY between motifs (after A ends, before B starts)
        between_start = a_start + len(cfg.motif_a)
        between_end = b_start
        gap_count = add_gaps_between(seq, between_start, between_end, cfg)

        record.update({
            "a_start": a_start,
            "b_start": b_start,
            "spacing": spacing,
            "gap_count": gap_count,
            "gap_region": [between_start, between_end],
        })

    elif case == "A_only":
        a_start = sample_position_single(cfg, "A")
        overwrite(seq, a_start, cfg.motif_a)
        record["a_start"] = a_start

    else:  # "B_only"
        b_start = sample_position_single(cfg, "B")
        overwrite(seq, b_start, cfg.motif_b)
        record["b_start"] = b_start

    record["sequence"] = "".join(seq)
    return record

def generate_dataset(cfg: SimConfig, n: int) -> List[Dict]:
    random.seed(cfg.seed)
    return [generate_one(cfg) for _ in range(n)]

def save_jsonl(records: List[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

def main():
    cfg = SimConfig()
    train = generate_dataset(cfg, n=20000)
    valid = generate_dataset(SimConfig(seed=2), n=2000)

    save_jsonl(train, "data/simulated_s1_gap_between/train.jsonl")
    save_jsonl(valid, "data/simulated_s1_gap_between/valid.jsonl")

    # save config for reproducibility
    Path("data/simulated_s1_gap_between").mkdir(parents=True, exist_ok=True)
    with open("data/simulated_s1_gap_between/config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    print("Saved:", len(train), len(valid))
    # quick sanity counts
    from collections import Counter
    c = Counter([r["case"] for r in train])
    print("Train case counts:", dict(c))

if __name__ == "__main__":
    main()

