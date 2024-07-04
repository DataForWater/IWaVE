import iwave
import sphinx_autosummary_accessors

# -- Project information -----------------------------------------------------

project = 'IWaVE'
copyright = '2024, Giulio Dolcetti, Salvador Pena-Haro, Hessel Winsemius'
author = 'Giulio Dolcetti, Salvador Pena-Haro, Hessel Winsemius'
release = iwave.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autosummary_accessors",
    "sphinx_design"
]
autosummary_generate = True
templates_path = ["_templates", sphinx_autosummary_accessors.templates_path]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = ".rst"
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
autodoc_member_order = "bysource"
autoclass_content = "both"

html_static_path = ['_static']
# No specific theme applied yet. Consider altering the theme css below.
# html_css_files = ["some-theme-iwave.css"]

html_theme_options = {
    "show_nav_level": 2,
    "navbar_align": "content",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/DataForWater/IWaVE",  # required
            "icon": "https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg",
            "type": "url",
        },
        {
            "name": "Data4Water",
            "url": "https://github.com/DataForWater",
            "icon": "_static/logo.jpg",
            "type": "local",
        },
    ],
}

html_context = {
    "github_url": "https://github.com",
    "github_user": "DataForWater",
    "github_repo": "IWaVE",
    "github_version": "docs",
    "doc_path": "docs",
}


remove_from_toctrees = ["_generated/*", "_build/doctrees/*"]

