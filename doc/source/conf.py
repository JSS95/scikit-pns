# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "scikit-pns"
copyright = "2025, Jisoo Song"
author = "Jisoo Song"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "numpydoc",
    "matplotlib.sphinxext.plot_directive",
]

autodoc_member_order = "bysource"

numpydoc_use_plots = True
numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False

plot_include_source = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/JSS95/scikit-pns",
    "logo": {
        "text": "scikit-pns",
    },
    "show_toc_level": 2,
}

plot_html_show_formats = False
plot_html_show_source_link = False


def autodoc_skip_member(app, what, name, obj, skip, options):
    """Skip metadata routing methods from scikit-learn."""
    exclude_methods = {
        "set_inverse_transform_request",
        "set_transform_request",
        "set_fit_request",
        "set_score_request",
        "set_partial_fit_request",
        "get_metadata_routing",
    }
    if name in exclude_methods:
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
