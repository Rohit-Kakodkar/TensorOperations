"""Sphinx configuration for the TensorOperations documentation."""

import subprocess
from pathlib import Path

docs_dir = Path(__file__).parent
doxygen_xml = docs_dir / "_build" / "doxygen" / "xml"

project = "TensorOperations"
copyright = "TensorOperations contributors"

extensions = ["myst_parser", "breathe"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build", "Doxyfile", "requirements.txt"]

# The guides link to headers and tests outside docs/ with relative paths;
# Sphinx publishes those as downloadable source files.

myst_heading_anchors = 4

html_theme = "furo"

breathe_projects = {"TensorOperations": str(doxygen_xml)}
breathe_default_project = "TensorOperations"
breathe_default_members = ("members",)


def _run_doxygen(app):
    doxygen_xml.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["doxygen", "Doxyfile"], cwd=docs_dir, check=True)


def setup(app):
    app.connect("builder-inited", _run_doxygen)
