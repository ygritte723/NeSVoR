# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

info_dict: dict = dict()
with open("../nesvor/version.py") as fp:
    exec(fp.read(), info_dict)

project = "NeSVoR"
copyright = info_dict["__copyright__"]
author = info_dict["__author__"]
release = info_dict["__version__"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinxarg.ext",
]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme

    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# html_static_path = ["_static"]
