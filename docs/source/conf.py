# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Mesh2vec"
copyright = "2023, Renumics GmbH"
author = "Renumics GmbH"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.doctest",
    "sphinx_gallery.gen_gallery",
    "sphinx.ext.napoleon",
    "sphinx.ext.extlinks",
]
autoclass_content = "class"
autosummary_generate = True
autodoc_default_options = {
    #'members': False,
    #'inherited-members': False,
    #'private-members': False,
    "special-members": "__init__",
    "member-order": "bysource",
}
autodoc_typehints = "description"

sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": [
        "generated_examples",
    ],  # path to where to save gallery generated output
    "within_subsection_order": "FileNameSortKey",
    "filename_pattern": "/",
    # directory where function/class granular galleries are stored
    "backreferences_dir": "generated_backreferences",
    # Modules for which function/class level galleries are created.
    "doc_module": ("renumics",),
}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {"logo_only": True}
html_css_files = ["custom.css"]

import plotly.io as pio

pio.renderers.default = "sphinx_gallery"
