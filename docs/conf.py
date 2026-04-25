# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "NGSDiffGeo"
copyright = "2025, Michael Neunteufel"
author = "Michael Neunteufel"
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "nbsphinx",
    "sphinx.ext.mathjax",  # Optional, for LaTeX support in notebooks
    "myst_parser",  # To parse markdown files
]

# Whether or not to evaluate the notebooks prior to embedding them
# evaluate_notebooks = True  # Default: True

# actually used by nbsphinx (evaluate_notebooks = True is ignored)
nbsphinx_execute = "always"  # or "auto" to skip existing output execution

# START nbsphinx stuff
# increase timeout for cell execution, since some files take long to execute
nbsphinx_timeout = 100000

# If True, the build process is continued even if an exception occurs:
nbsphinx_allow_errors = False


templates_path = ["_templates"]


def _notebooks_not_listed_in_index():
    """Exclude notebooks that are present in docs but not linked from index.rst."""
    from pathlib import Path

    docs_dir = Path(__file__).parent
    index_text = (docs_dir / "index.rst").read_text(encoding="utf-8")

    listed_notebooks = {
        (docs_dir / line.strip()).resolve()
        for line in index_text.splitlines()
        if line.strip().endswith(".ipynb")
    }

    return [
        notebook.relative_to(docs_dir).as_posix()
        for notebook in docs_dir.rglob("*.ipynb")
        if "_build" not in notebook.parts and notebook.resolve() not in listed_notebooks
    ]


exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    *_notebooks_not_listed_in_index(),
]


def _notebooks_not_listed_in_index():
    """Exclude notebooks that are present in docs but not linked from index.rst."""
    from pathlib import Path

    docs_dir = Path(__file__).parent
    index_text = (docs_dir / "index.rst").read_text(encoding="utf-8")

    listed_notebooks = {
        (docs_dir / line.strip()).resolve()
        for line in index_text.splitlines()
        if line.strip().endswith(".ipynb")
    }

    return [
        notebook.relative_to(docs_dir).as_posix()
        for notebook in docs_dir.rglob("*.ipynb")
        if "_build" not in notebook.parts and notebook.resolve() not in listed_notebooks
    ]


exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    *_notebooks_not_listed_in_index(),
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

nbsphinx_widgets_path = "https://cdn.jsdelivr.net/npm/@jupyter-widgets/html-manager@^1.0.1/dist/embed-amd.js"
nbsphinx_widgets_options = {"priority": 500}
nbsphinx_requirejs_path = (
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"
)
nbsphinx_requirejs_options = {
    "priority": 400,
}

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


def setup(app):
    app.add_js_file("webgui_jupyter_widgets.js", priority=450)
