import os
import re
import json
import argparse
from collections import defaultdict

from neo4j import GraphDatabase


def safe(name):
    """Make a label / relationship type safe to put in a Cypher query."""
    return re.sub(r"\W+", "_", str(name)).strip("_") or "Unknown"


def flat(props):
    out = {}
    for k, v in (props or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[safe(k)] = v
        elif isinstance(v, list) and all(isinstance(x, (str, int, float, bool)) for x in v):
            out[safe(k)] = v
        else:
            out[safe(k)] = json.dumps(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file")
    args = ap.parse_args()

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise SystemExit("Set the NEO4J_PASSWORD environment variable.")

    data = json.load(open(args.json_file, encoding="utf-8"))
    doi = data.get("doi")
    if not doi:
        raise SystemExit("JSON has no DOI; re-run extract.py with --doi.")

    # Paper node + all other nodes, grouped by label
    by_label = defaultdict(list)
    by_label["Paper"].append({"uid": doi, "name": doi, "props": {"doi": doi}})

    label_of = {}
    for n in data["nodes"]:
        label = safe(n["label"])
        label_of[n["id"]] = label
        props = flat(n.get("properties"))
        props["doi"] = doi
        by_label[label].append({"uid": f"{doi}::{n['id']}", "name": n["name"], "props": props})

    # Relationships grouped by (type, source label, target label)
    by_rel = defaultdict(list)
    for r in data["relationships"]:
        if r["source"] not in label_of or r["target"] not in label_of:
            continue
        key = (safe(r["type"]).upper(), label_of[r["source"]], label_of[r["target"]])
        by_rel[key].append({
            "s": f"{doi}::{r['source']}",
            "t": f"{doi}::{r['target']}",
            "props": flat(r.get("properties")),
        })

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        for label, rows in by_label.items():
            session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.uid IS UNIQUE")
            session.run(
                f"UNWIND $rows AS row MERGE (n:`{label}` {{uid: row.uid}}) "
                f"SET n += row.props, n.name = row.name",
                rows=rows,
            )
            print(f"{len(rows):>4} {label}")

        for (rtype, sl, tl), rows in by_rel.items():
            session.run(
                f"UNWIND $rows AS row "
                f"MATCH (a:`{sl}` {{uid: row.s}}) MATCH (b:`{tl}` {{uid: row.t}}) "
                f"MERGE (a)-[r:`{rtype}`]->(b) SET r += row.props",
                rows=rows,
            )
            print(f"{len(rows):>4} (:{sl})-[:{rtype}]->(:{tl})")
    driver.close()


if __name__ == "__main__":
    main()