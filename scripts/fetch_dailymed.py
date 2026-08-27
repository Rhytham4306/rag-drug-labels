"""
Fetch real FDA drug label (Structured Product Labeling) text from DailyMed's
public REST API — free, no API key required.

API docs: https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm

Usage:
    python scripts/fetch_dailymed.py --drugs ibuprofen metformin lisinopril --out data/dailymed
"""
import argparse
import re
import time
from pathlib import Path

import requests

BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"


def search_set_id(drug_name: str) -> str | None:
    """Look up the SPL 'setid' for a drug name via DailyMed's search endpoint."""
    resp = requests.get(
        f"{BASE_URL}/spls.json",
        params={"drug_name": drug_name, "pagesize": 1},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("data", [])
    if not results:
        return None
    return results[0]["setid"]


def fetch_spl_text(setid: str) -> str:
    """Fetch the human-readable SPL document text for a given setid."""
    resp = requests.get(f"{BASE_URL}/spls/{setid}.xml", timeout=30)
    resp.raise_for_status()
    xml = resp.text
    # Strip XML tags to get readable text; good enough for chunking purposes.
    # (For production use, a proper SPL XML parser would preserve structure
    # more precisely — noted as a possible improvement in the README.)
    text = re.sub(r"<[^>]+>", "\n", xml)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Fetch drug labels from DailyMed")
    parser.add_argument("--drugs", nargs="+", required=True, help="Drug names to fetch")
    parser.add_argument("--out", default="data/dailymed", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for drug in args.drugs:
        print(f"Searching DailyMed for '{drug}'...")
        try:
            setid = search_set_id(drug)
        except requests.RequestException as e:
            print(f"  Network error for {drug}: {e}")
            continue

        if not setid:
            print(f"  No results found for '{drug}', skipping.")
            continue

        print(f"  Found setid {setid}, fetching label text...")
        try:
            text = fetch_spl_text(setid)
        except requests.RequestException as e:
            print(f"  Failed to fetch label for {drug}: {e}")
            continue

        out_path = out_dir / f"{drug.lower().replace(' ', '_')}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"  Saved -> {out_path}")

        time.sleep(1)  # be polite to the free public API

    print(f"\nDone. Labels saved in {out_dir}/")
    print("Next: python -m src.vectorstore --docs", out_dir, "--persist ./chroma_db")


if __name__ == "__main__":
    main()
