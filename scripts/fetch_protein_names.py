import sys
import time
from pathlib import Path

import pandas as pd
import requests

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from utils import DATA

DATASETS = ["kiba", "davis", "papyrus", "bindingdb"]
ENDPOINT = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = "accession,protein_name,gene_primary,organism_name,reviewed,sequence"
BATCH = 40
PAUSE = 0.2
PREFERRED_ORGANISM = "Homo sapiens"


def crc64(text):
    """UniProt's sequence checksum: CRC64-ISO, reflected, polynomial 0xd800000000000000."""
    poly, table = 0xd800000000000000, []
    for index in range(256):
        part = index
        for _ in range(8):
            part = (part >> 1) ^ poly if part & 1 else part >> 1
        table.append(part)

    crc = 0
    for character in text:
        crc = table[(crc ^ ord(character)) & 0xFF] ^ (crc >> 8)

    return f"{crc:016X}"


def rank(entry):
    """Reviewed entries first, then the organism the panels are drawn from."""
    reviewed = entry.get("entryType", "").startswith("UniProtKB reviewed")
    organism = entry.get("organism", {}).get("scientificName", "")

    return (not reviewed, organism != PREFERRED_ORGANISM, entry["primaryAccession"])


def describe(entry):
    """(gene symbol, full protein name) for one UniProt entry, each possibly empty."""
    genes = entry.get("genes") or [{}]
    gene = genes[0].get("geneName", {}).get("value", "")
    description = entry.get("proteinDescription", {})
    named = description.get("recommendedName") or (description.get("submissionNames") or [{}])[0]

    return gene, named.get("fullName", {}).get("value", "")


def lookup(checksums, session):
    """{checksum: best entry} for one batch of sequence checksums."""
    query = " OR ".join(f"checksum:{value}" for value in checksums)
    response = session.get(ENDPOINT, params={"query": f"({query})", "fields": FIELDS,
                                                "size": 500}, timeout=60)
    response.raise_for_status()

    best = {}
    for entry in response.json().get("results", []):
        checksum = entry.get("sequence", {}).get("crc64")
        if checksum in checksums and (checksum not in best or rank(entry) < rank(best[checksum])):
            best[checksum] = entry

    return best


def fetch(dataset, session):
    """Refresh one dataset's protein_names.csv, fetching only the ids it does not have."""
    proteins = pd.read_csv(DATA / dataset / "proteins.csv")
    path = DATA / dataset / "protein_names.csv"
    known = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["protein_id"])
    todo = proteins[~proteins["protein_id"].isin(known["protein_id"])]

    if todo.empty:
        print(f"{dataset:10s} {len(known)} names cached, nothing to fetch")
        return known

    todo = todo.assign(checksum=[crc64(sequence) for sequence in todo["target_sequence"]])
    rows, unique = [], sorted(set(todo["checksum"]))
    print(f"{dataset:10s} {len(todo)} proteins, {len(unique)} distinct sequences")

    found = {}
    for start in range(0, len(unique), BATCH):
        found.update(lookup(unique[start:start + BATCH], session))
        print(f"  {min(start + BATCH, len(unique))}/{len(unique)} looked up", flush=True)
        time.sleep(PAUSE)

    for row in todo.itertuples():
        entry = found.get(row.checksum)
        gene, name = describe(entry) if entry else ("", "")
        rows.append({"protein_id": row.protein_id,
                        "accession": entry["primaryAccession"] if entry else "",
                        "gene": gene, "protein_name": name,
                        "organism": entry.get("organism", {}).get("scientificName", "") if entry else ""})

    table = pd.concat([known, pd.DataFrame(rows)], ignore_index=True).sort_values("protein_id")
    table.to_csv(path, index=False)
    named = int((table["gene"].fillna("") != "").sum())
    print(f"{dataset:10s} {named}/{len(table)} named -> {path}")

    return table


def main():
    session = requests.Session()
    for dataset in (sys.argv[1:] or DATASETS):
        fetch(dataset, session)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
