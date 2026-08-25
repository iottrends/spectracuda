"""Builds the published book (see README.md's "The book" section) from
book_template.html by base64-embedding this directory's own *.png
figures in place of their __IMG_*__ placeholders.

Usage: .venv/bin/python docs/book_figures/build_book.py
Requires the figures to already exist -- run generate_figures.py first
if any are missing or stale:
    .venv/bin/python docs/book_figures/generate_figures.py
    .venv/bin/python docs/book_figures/build_book.py

Output: docs/book_figures/book.html -- self-contained (images inlined as
data URIs), ready to hand to the Artifact tool to publish/republish. Not
imported by spectracuda itself; this whole directory is a docs-only tool
(see the `docs` extra in pyproject.toml).
"""
import base64
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

placeholders = {
    "__IMG_SYNC_METRIC__": "sync_metric.png",
    "__IMG_CFO_ROTATION__": "cfo_rotation.png",
    "__IMG_CFO_RECOVERED__": "cfo_recovered.png",
    "__IMG_CHANNEL_EQ__": "channel_equalization.png",
    "__IMG_CONSTELLATION__": "constellation_gallery.png",
    "__IMG_FEC_COMPARISON__": "fec_comparison.png",
}

with open(os.path.join(THIS_DIR, "book_template.html")) as f:
    content = f.read()

for placeholder, filename in placeholders.items():
    path = os.path.join(THIS_DIR, filename)
    with open(path, "rb") as imgf:
        b64 = base64.b64encode(imgf.read()).decode("ascii")
    data_uri = f"data:image/png;base64,{b64}"
    count = content.count(placeholder)
    assert count == 1, f"{placeholder} appears {count} times, expected 1"
    content = content.replace(placeholder, data_uri)

out_path = os.path.join(THIS_DIR, "book.html")
with open(out_path, "w") as f:
    f.write(content)

size_mb = os.path.getsize(out_path) / (1024 * 1024)
print(f"wrote {out_path} ({size_mb:.2f} MB)")
