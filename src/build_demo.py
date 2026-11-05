"""Inject results/uci_LDA_demo.json into demo/template.html -> docs/index.html.

docs/index.html is a single self-contained page, so it works on GitHub Pages
or opened straight from disk.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "results" / "uci_LDA_demo.json").read_text())

slim = {"gestures": d["gestures"], "alphas": d["alphas"],
        "increment_ms": d["increment_ms"], "people": {}}
for pid, p in d["people"].items():
    slim["people"][pid] = {k: p[k] for k in
                           ("y", "pred", "conf", "accuracy", "conformal+personal")}

html = (ROOT / "demo" / "template.html").read_text()
html = html.replace("__DATA__", json.dumps(slim, separators=(",", ":")))
out = ROOT / "docs" / "index.html"
out.parent.mkdir(exist_ok=True)
out.write_text(html)
print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} KB)")
