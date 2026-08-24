"""Sphinx configuration for spectracuda's docs (built by Read the Docs at
spectracuda.readthedocs.io).

Source is Markdown throughout, parsed via MyST -- every existing docs/*.md
file (architecture.md, mac.md, ldpc.md, todo.md,
liquid-dsp-api-inventory.md) is picked up here completely unchanged, no
reStructuredText rewrite. The book/ subdirectory holds the narrative
"OFDM Field Guide" chapters ported from docs/book_figures/book_template.html
(see book/README.md for why two forms of that content exist side by side).
"""
from __future__ import annotations

project = "spectracuda"
copyright = "2026, abhinav"
author = "abhinav"

extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.mathjax",
]

source_suffix = {
    ".md": "markdown",
}

# colon_fence: lets callouts use ```{note}/{warning}``` directives (used for
# the book's "a real deviation from the textbook formula" / "an honest,
# measured limit" style callouts). deflist: not required by current content
# but cheap to enable for future pages.
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

root_doc = "index"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

html_theme = "sphinx_rtd_theme"
html_title = "spectracuda"

# "none": several existing docs/*.md files use bare ``` fences for ASCII
# diagrams (box-drawing characters, arrows), not code -- defaulting to
# "python" made Sphinx try to lex those as Python and warn. Snippets that
# ARE Python already say so explicitly (```python) throughout.
highlight_language = "none"
