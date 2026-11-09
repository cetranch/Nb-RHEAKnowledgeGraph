import sys
import json
import random
import argparse
from pathlib import Path

import ollama
from pypdf import PdfReader

MODEL = "qwen2.5:7b"

CHUNK_CHARS = 6000     # bigger than for tuples: the model needs context to judge the ontology
MAX_EPOCHS = 6
STOP_THRESHOLD = 2

SCHEMA = {
    "type": "object",
    "properties": {
        "classes": {
            "type": "array",
            "maxItems": 30,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "maxLength": 60},
                    "description": {"type": "string", "maxLength": 150},
                },
                "required": ["name", "description"],
            },
        },
        "relations": {
            "type": "array",
            "maxItems": 50,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "maxLength": 60},
                    "domain": {"type": "string", "maxLength": 60},
                    "range": {"type": "string", "maxLength": 60},
                },
                "required": ["name", "domain", "range"],
            },
        },
    },
    "required": ["classes", "relations"],
}

GOAL = (
    "We are building a knowledge graph from papers on NIOBIUM ALLOYS to understand how the "
    "alloy's composition (elements, amounts, processing) affects OXIDATION BEHAVIOR "
    "(mass gain, oxide scale, kinetics, pest oxidation, temperature/atmosphere, etc.)."
)

RULES = (
    "Rules:\n"
    "- Classes are TYPES (e.g. Alloy, Element, Specimen, OxidationTest, Oxide, Property, Phase). "
    "Specific instances like 'Nb-10W-1Zr' or '1200 C' are NOT classes.\n"
    "- Class names: singular PascalCase. Relation names: lower_snake_case verbs (e.g. has_component).\n"
    "- Every relation's domain and range must be a class in the ontology.\n"
    "- Prefer reusing or generalizing existing classes over adding near-duplicates.\n"
    "- Only change the ontology if the text needs it. Keep it compact (roughly 10-25 classes).\n"
    "- Return the COMPLETE updated ontology, including everything that did not change."
)


def extract_pdf_text(pdf_path) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join((p.extract_text() or "") for p in reader.pages).strip()


def chunk_text(text: str, size: int = CHUNK_CHARS):
    paragraphs = text.split("\n")
    chunk, chunks = "", []
    for para in paragraphs:
        if len(chunk) + len(para) > size and chunk:
            chunks.append(chunk)
            chunk = ""
        chunk += para + "\n"
    if chunk.strip():
        chunks.append(chunk)
    return chunks


def ontology_to_text(onto: dict) -> str:
    if not onto["classes"]:
        return "(empty - create the initial ontology from the text)"
    lines = ["CLASSES:"]
    lines += [f"- {c['name']}: {c['description']}" for c in onto["classes"]]
    lines.append("RELATIONS:")
    lines += [f"- {r['name']}({r['domain']} -> {r['range']})" for r in onto["relations"]]
    return "\n".join(lines)


def update_ontology(onto: dict, chunk: str) -> dict:
    prompt = (
        f"{GOAL}\n\n{RULES}\n\n"
        f"CURRENT ONTOLOGY:\n{ontology_to_text(onto)}\n\n"
        f"TEXT:\n{chunk}"
    )
    new = None
    for attempt in range(3):
        resp = ollama.generate(
            model=MODEL,
            prompt=prompt,
            format=SCHEMA,
            think=False,
            # temperature 0 can loop forever; add a little randomness on retries
            options={"temperature": 0.1 * attempt, "num_predict": 4096, "num_ctx": 16384,
                     "repeat_penalty": 1.1},
        )
        try:
            new = json.loads(resp.response)
            break
        except json.JSONDecodeError as e:
            print(f"  Failed to parse JSON (attempt {attempt + 1}/3): {e}")
    if new is None:
        return onto

    # Remove duplicate classes / relations the model may have repeated
    seen, classes = set(), []
    for c in new["classes"]:
        if c["name"].lower() not in seen:
            seen.add(c["name"].lower())
            classes.append(c)
    new["classes"] = classes
    seen, rels = set(), []
    for r in new["relations"]:
        key = (r["name"].lower(), r["domain"].lower(), r["range"].lower())
        if key not in seen:
            seen.add(key)
            rels.append(r)
    new["relations"] = rels

    # Keep only relations whose domain/range are real classes
    names = {c["name"] for c in new["classes"]}
    new["relations"] = [r for r in new["relations"] if r["domain"] in names and r["range"] in names]

    # Guard against the model wiping the ontology out
    if onto["classes"] and len(new["classes"]) < 0.7 * len(onto["classes"]):
        print("  Rejected update: it dropped too many classes")
        return onto
    return new


def signature(onto: dict) -> set:
    """Set of things that count as 'the ontology' for measuring change."""
    s = {("class", c["name"].lower()) for c in onto["classes"]}
    s |= {("rel", r["name"].lower(), r["domain"].lower(), r["range"].lower()) for r in onto["relations"]}
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_folder")
    parser.add_argument("--out", default="ontology.json")
    args = parser.parse_args()

    pdfs = sorted(Path(args.pdf_folder).glob("*.pdf"))
    if not pdfs:
        print("No PDFs found.")
        sys.exit(1)
    print(f"Found {len(pdfs)} PDFs")

    onto = {"classes": [], "relations": []}

    # --- Seed from the first paper ---
    print(f"\nSeeding from {pdfs[0].name}")
    for chunk in chunk_text(extract_pdf_text(pdfs[0])):
        onto = update_ontology(onto, chunk)
    print(f"Seed: {len(onto['classes'])} classes, {len(onto['relations'])} relations")

    # --- Epochs over all papers ---
    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n===== EPOCH {epoch} =====")
        before = signature(onto)

        order = pdfs[:]
        random.shuffle(order)
        for pdf in order:
            text = extract_pdf_text(pdf)
            if not text:
                print(f"  {pdf.name}: no text, skipping")
                continue
            for chunk in chunk_text(text):
                onto = update_ontology(onto, chunk)
            print(f"  {pdf.name}: {len(onto['classes'])} classes, {len(onto['relations'])} relations")
            Path(args.out).write_text(json.dumps(onto, indent=2))   # checkpoint

        changes = len(before ^ signature(onto))
        print(f"Epoch {epoch}: {changes} class/relation change(s)")
        if changes <= STOP_THRESHOLD:
            print("Converged.")
            break

    print(f"\nFinal ontology saved to {args.out}")
    print(ontology_to_text(onto))


if __name__ == "__main__":
    main()