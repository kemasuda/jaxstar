from importlib.metadata import version

import jaxstar


def test_package_version_matches_installed_metadata():
    assert jaxstar.__version__ == version("jaxstar")
