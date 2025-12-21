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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

nbsphinx_widgets_path = (
    "https://unpkg.com/@jupyter-widgets/html-manager@*/dist/embed-amd.js"
)
nbsphinx_requirejs_path = (
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js"
)

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
