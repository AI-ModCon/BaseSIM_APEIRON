"""Sphinx configuration for the Apeiron documentation.

The heavy runtime dependencies (torch, river, evidently, wandb, ...) are mocked
via ``autodoc_mock_imports`` so the docs build stays fast and does not need a
GPU-specific PyTorch wheel. Only ``docs/requirements.txt`` is installed.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

# Make the installable package under src/ importable for autodoc.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# -- Project information -----------------------------------------------------

project = "Apeiron"
author = "AI-ModCon"
copyright = f"{date.today().year}, AI-ModCon"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

exclude_patterns = [
    "_build",
    "README.md",
    "requirements.txt",
    "Thumbs.db",
    ".DS_Store",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

# -- MyST ---------------------------------------------------------------------

myst_enable_extensions = [
    "attrs_inline",
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
]
# Auto-generate anchors for headings so `file.md#some-heading` links resolve.
myst_heading_anchors = 3

# -- autodoc ------------------------------------------------------------------

autodoc_mock_imports = [
    "torch",
    "torchvision",
    "transformers",
    "river",
    "evidently",
    "wandb",
    "mlflow",
    "matplotlib",
    "psutil",
    "pynvml",
    "nvidia_ml_py",
]

autodoc_default_options = {
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# Mocked modules produce unresolvable type targets; do not fail the build on them.
nitpicky = False

# -- HTML output --------------------------------------------------------------

html_theme = "furo"
html_title = f"{project} {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#0a4bff",
        "color-brand-content": "#2757dd",
    },
    "dark_css_variables": {
        "color-brand-primary": "#3d94ff",
        "color-brand-content": "#5ca5ff",
    },
    "source_repository": "https://github.com/AI-ModCon/BaseSIM_APEIRON/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/AI-ModCon/BaseSIM_APEIRON",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 '
                "8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 "
                "0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 "
                "1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 "
                "0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 "
                "1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 "
                "2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 "
                '1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}
