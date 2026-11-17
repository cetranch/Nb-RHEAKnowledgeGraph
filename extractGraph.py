import re
import sys
import json
import argparse
from pathlib import Path

import ollama
from pypdf import PdfReader

MODEL = "qwen3:30b"
CHUNK_CHARS = 6000

SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "name": {"type": "string"},
                    "properties": {"type": "object"},
                },
                "required": ["id", "label", "name"],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "type": {"type": "string"},
                    "target": {"type": "string"},
                    "properties": {"type": "object"},
                },
                "required": ["source", "type", "target"],
            },
        },
    },
    "required": ["nodes", "relationships"],
}

FORMAT_INSTRUCTIONS = """

# OUTPUT FORMAT (added automatically)
This is one excerpt of a larger paper. Extract only what appears in this excerpt.
Return JSON: {"nodes": [{"id", "label", "name", "properties"}], "relationships": [{"source", "type", "target", "properties"}]}
- "label" and "type" must come from the ontology above.
- "source" and "target" are node ids.
- Use stable, readable ids (e.g. "elem_Nb") so the same entity gets the same id in every excerpt.
- Properties must be flat: strings, numbers, booleans, or lists of those.
- Do not create a node for the paper itself; that is added automatically.
"""


def read_text(path):
    if path.lower().endswith(".pdf"):
        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages).strip()
    return Path(path).read_text(encoding="utf-8", errors="ignore").strip()


def find_doi(text):
    m = re.search(r"\b(10\.\d{4,9}/[^\s\"<>]+)", text[:15000])
    return m.group(1).rstrip(".,;)") if m else None


def chunk_text(text, size=CHUNK_CHARS):
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) > size and cur:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        chunks.append(cur)
    return chunks


def ask(prompt, model):
    resp = ollama.generate(
        model=model,
        prompt=prompt,
        format=SCHEMA,
        think=False,
        options={"temperature": 0, "num_ctx": 16384, "num_predict": 4096},
    )
    try:
        return json.loads(resp.response)
    except json.JSONDecodeError:
        print("  (bad JSON from model, skipping chunk)")
        return {"nodes": [], "relationships": []}


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paper", help="PDF or .txt file")
    ap.add_argument("prompt_file", help="Prompt containing your ontology and rules")
    ap.add_argument("-o", "--output", help="Output JSON path (default: <paper>.json)")
    ap.add_argument("--doi", help="DOI (default: auto-detected from text)")
    args = ap.parse_args()

    template = Path(args.prompt_file).read_text(encoding="utf-8")
    text = read_text(args.paper)
    if not text:
        sys.exit("No extractable text found.")

    doi = args.doi or find_doi(text)
    print(f"DOI: {doi or 'NOT FOUND (use --doi)'}")

    chunks = chunk_text(text)
    print(f"{len(chunks)} chunk(s)")

    nodes = {}  # (label, normalized name) -> node
    rels = {}   # (source id, type, target id) -> relationship

    for i, chunk in enumerate(chunks, 1):
        prompt = template.replace("{{DOI}}", doi or "null") + FORMAT_INSTRUCTIONS + "\n\nTEXT:\n" + chunk
        data = ask(prompt, MODEL)

        idmap = {}
        for n in data.get("nodes", []):
            key = (n["label"], norm(n["name"]))
            if key not in nodes:
                nodes[key] = {
                    "id": n["id"], "label": n["label"], "name": n["name"],
                    "properties": n.get("properties", {}),
                }
            else:  # merge: keep existing values, add new ones
                for k, v in n.get("properties", {}).items():
                    nodes[key]["properties"].setdefault(k, v)
            idmap[n["id"]] = nodes[key]["id"]

        for r in data.get("relationships", []):
            s, t = idmap.get(r["source"]), idmap.get(r["target"])
            if s and t:
                rels.setdefault((s, r["type"], t), {
                    "source": s, "type": r["type"], "target": t,
                    "properties": r.get("properties", {}),
                })

        print(f"  chunk {i}/{len(chunks)}: {len(data.get('nodes', []))} nodes, "
              f"{len(data.get('relationships', []))} rels")

    result = {
        "doi": doi,
        "nodes": list(nodes.values()),
        "relationships": list(rels.values()),
    }
    out = args.output or str(Path(args.paper).with_suffix(".json"))
    Path(out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}: {len(result['nodes'])} nodes, {len(result['relationships'])} relationships")


if __name__ == "__main__":
    main()
