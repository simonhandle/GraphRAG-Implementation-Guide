"""Render the GraphRAG knowledge graph (output/*.parquet) as an interactive HTML page.

Usage:
    python visualize_graph.py            # writes graph.html
    # in the notebook:
    from visualize_graph import show; show()
"""

import html
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
OUTPUT = ROOT / "output"

PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
           "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]
NO_COMMUNITY = "#cccccc"


def build_html(level: int = 0, hide_isolated: bool = False) -> str:
    entities = pd.read_parquet(OUTPUT / "entities.parquet")
    relationships = pd.read_parquet(OUTPUT / "relationships.parquet")
    communities = pd.read_parquet(OUTPUT / "communities.parquet")

    # entity id -> community number at the chosen hierarchy level
    membership = {}
    for _, c in communities[communities["level"] == level].iterrows():
        for eid in c["entity_ids"]:
            membership[eid] = int(c["community"])
    community_ids = sorted(set(membership.values()))
    colors = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(community_ids)}

    connected = set(relationships["source"]) | set(relationships["target"])
    nodes = []
    for _, e in entities.iterrows():
        if hide_isolated and e["title"] not in connected:
            continue
        community = membership.get(e["id"])
        nodes.append({
            "id": e["title"],
            "label": e["title"],
            "value": int(e["degree"]) + 1,
            "color": colors.get(community, NO_COMMUNITY),
            "title": f"{e['title']} ({e['type']})\n"
                     f"Community: {community if community is not None else '–'} · degree {e['degree']}\n\n"
                     f"{e['description']}",
        })

    edges = [{
        "from": r["source"],
        "to": r["target"],
        "value": float(r["weight"]),
        "title": f"{r['source']} → {r['target']}\n\n{r['description']}",
    } for _, r in relationships.iterrows()]

    legend = "".join(
        f'<span><i style="background:{colors[c]}"></i>Community {c}</span>' for c in community_ids
    ) + f'<span><i style="background:{NO_COMMUNITY}"></i>no community</span>'

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>GraphRAG Knowledge Graph</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  body {{ margin:0; font-family:system-ui,sans-serif; background:#fff; color:#222; }}
  #head {{ padding:8px 12px; font-size:13px; border-bottom:1px solid #ddd; }}
  #head span {{ margin-right:14px; white-space:nowrap; }}
  #head i {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:4px; }}
  #graph {{ height:calc(100vh - 40px); }}
  div.vis-tooltip {{ max-width:380px; white-space:pre-wrap; font-size:12px; }}
</style></head>
<body>
<div id="head"><b>{len(nodes)} entities · {len(edges)} relationships</b> &nbsp; {legend}</div>
<div id="graph"></div>
<script>
  const nodes = new vis.DataSet({json.dumps(nodes)});
  const edges = new vis.DataSet({json.dumps(edges)});
  new vis.Network(document.getElementById("graph"), {{nodes, edges}}, {{
    nodes: {{ shape: "dot", scaling: {{ min: 8, max: 40, label: {{ enabled: true, min: 11, max: 22 }} }} }},
    edges: {{ color: {{ color: "#aaa", highlight: "#333" }}, scaling: {{ min: 1, max: 6 }}, smooth: false }},
    physics: {{ solver: "forceAtlas2Based", stabilization: {{ iterations: 300 }} }},
    interaction: {{ hover: true, tooltipDelay: 100 }}
  }});
</script>
</body></html>"""


def write_html(path: Path = ROOT / "graph.html", **kwargs) -> Path:
    path.write_text(build_html(**kwargs), encoding="utf-8")
    return path


def show(height: int = 650, **kwargs):
    """Display the graph inline in a Jupyter / VS Code notebook."""
    from IPython.display import HTML, display
    page = html.escape(build_html(**kwargs), quote=True)
    display(HTML(f'<iframe srcdoc="{page}" style="width:100%;height:{height}px;border:0"></iframe>'))


if __name__ == "__main__":
    print(f"Wrote {write_html()}")
