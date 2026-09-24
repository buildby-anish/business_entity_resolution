"""Generate a small FAKE dataset in the PS layout so you can smoke-test the whole pipeline.
It is NOT the real challenge data and its noise is only a rough imitation.

    python scripts/make_synthetic_data.py --out dataset_fake --n-train 1500 --n-test 500
"""
import argparse
import csv
import random
from pathlib import Path

W1 = ["sharma", "gupta", "patel", "blue", "ocean", "zenith", "sunrise", "apex", "royal", "green", "star", "global",
      "metro", "prime", "delta", "omega", "lotus", "tiger", "eagle", "summit", "maple", "cedar", "harbor", "pioneer",
      "dupont", "martin", "bernard", "moreau", "iyer", "reddy", "khan", "nair"]
KIND = ["logistics", "traders", "hardware", "bakery", "cafe", "motors", "textiles", "pharma", "foods",
        "electronics", "constructions", "services", "transport", "printing"]
SUFFIX = {"India": ["Pvt Ltd", "Private Limited", "Pvt. Ltd."], "US": ["LLC", "Inc", "Corp", "Corporation"],
          "France": ["SARL", "SAS", "SA"]}
CITY = {"India": ["Pune", "Mumbai", "Nashik", "Nagpur", "Thane"], "US": ["Austin", "Boston", "Denver", "Seattle", "Miami"],
        "France": ["Paris", "Lyon", "Nantes", "Lille"]}
STREET = ["Main", "Oak", "Park", "Station", "Market", "Temple", "Lake", "Hill", "Mill", "Church", "Garden", "Nehru"]


def typo(s, rng):
    if len(s) < 5:
        return s
    i = rng.randrange(1, len(s) - 1)
    return rng.choice([s[:i] + s[i + 1:], s[:i] + s[i + 1] + s[i] + s[i + 2:]])


def make_entity(country, rng):
    w1, w2, kind = rng.choice(W1), rng.choice(W1), rng.choice(KIND)
    base = f"{w1.title()} {kind.title()}" if rng.random() < 0.5 else f"{w1.title()} {w2.title()} {kind.title()}"
    city = rng.choice(CITY[country])
    num = rng.randint(1, 400)
    street = rng.choice(STREET)
    if country == "India":
        addr = dict(num=f"Plot {num}", street=f"{street} Road", city=city, extra="Maharashtra", code=str(rng.randint(400000, 445999)))
    elif country == "US":
        addr = dict(num=str(num), street=f"{street} Street", city=city, extra="TX", code=str(rng.randint(10000, 99999)))
    else:
        addr = dict(num=str(num), street=f"Rue {street}", city=city, extra="", code=str(rng.randint(10000, 95999)))
    return dict(base=base, suffix=rng.choice(SUFFIX[country]), addr=addr, country=country)


def noisy_name(e, rng):
    name = e["base"]
    if rng.random() < 0.7:
        name += " " + e["suffix"]
    if rng.random() < 0.15:
        name = " ".join(reversed(name.split(" ", 1)[0:1] + [name.split(" ", 1)[-1]])) if " " in name else name
    if rng.random() < 0.25:
        name = typo(name, rng)
    if rng.random() < 0.15:
        name = name.upper()
    if rng.random() < 0.1:
        name = name.replace(" ", " & ", 1) if rng.random() < 0.5 else name
    return name


def noisy_addr(e, rng):
    a = dict(e["addr"])
    street = a["street"]
    if rng.random() < 0.5:
        street = street.replace("Road", "Rd").replace("Street", "St")
    parts = [a["num"], street]
    if rng.random() < 0.15:
        parts.append("Near SBI ATM")
    if rng.random() < 0.8:
        parts.append(a["city"])
    if a["extra"] and rng.random() < 0.5:
        parts.append(a["extra"])
    if rng.random() < 0.6:
        parts.append(a["code"])
    if rng.random() < 0.15:
        parts = parts[::-1]
    if rng.random() < 0.05:
        return ""
    return ", ".join(parts)


def write_tsv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n", escapechar="\\")
        w.writerow(header)
        w.writerows(rows)


def build(n, countries, rng, out_dir, prefix, with_truth):
    out_dir.mkdir(parents=True, exist_ok=True)
    s1, others, truth = [], {"S2": [], "S3": []}, {}
    for i in range(1, n + 1):
        c = rng.choice(countries)
        e = make_entity(c, rng)
        sid = f"S1-{i:05d}"
        s1.append((sid, f"{e['base']} {e['suffix']}", ", ".join([e["addr"]["num"], e["addr"]["street"], e["addr"]["city"], e["addr"]["code"]]), c))
        truth[sid] = []
        if rng.random() < 0.25:                      # singleton: no matches, but a look-alike trap in another city
            if rng.random() < 0.7:
                t = make_entity(c, rng); t["base"] = e["base"]
                others[rng.choice(["S2", "S3"])].append((noisy_name(t, rng), noisy_addr(t, rng), c, None))
            continue
        for _ in range(rng.choice([1, 1, 2, 3])):
            src = rng.choice(["S2", "S3"])
            others[src].append((noisy_name(e, rng), noisy_addr(e, rng), c, sid))
        if rng.random() < 0.3:                        # look-alike but different business (hard negative)
            t = make_entity(c, rng); t["base"] = e["base"]
            others[rng.choice(["S2", "S3"])].append((noisy_name(t, rng), noisy_addr(t, rng), c, None))
    for _ in range(int(0.3 * n)):                     # orphans
        c = rng.choice(countries)
        e = make_entity(c, rng)
        others[rng.choice(["S2", "S3"])].append((noisy_name(e, rng), noisy_addr(e, rng), c, None))
    hdr = ["entity_id", "business_name", "business_address", "country"]
    write_tsv(out_dir / f"{prefix}_source1.tsv", hdr, s1)
    for src in ("S2", "S3"):
        rows = others[src]
        rng.shuffle(rows)
        recs = []
        for j, (nm, ad, c, owner) in enumerate(rows, 1):
            rid = f"{src}-{j:05d}"
            recs.append((rid, nm, ad, c))
            if owner:
                truth[owner].append(rid)
        write_tsv(out_dir / f"{prefix}_source{src[1]}.tsv", hdr, recs)
    if with_truth:
        write_tsv(out_dir / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"],
                  [(s, ",".join(sorted(v))) for s, v in truth.items()])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="dataset_fake")
    p.add_argument("--n-train", type=int, default=1500)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    root = Path(a.out)
    build(a.n_train, ["India", "US"], random.Random(a.seed), root / "train", "train", True)
    build(a.n_test, ["India", "US", "France"], random.Random(a.seed + 1), root / "test", "test", False)   # France only in test
    print("wrote", root)
