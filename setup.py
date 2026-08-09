#!/usr/bin/env python
import codecs
import os
import re
from setuptools import find_namespace_packages, setup

# PROJECT SPECIFIC
NAME = "jaxstar"
PACKAGES = find_namespace_packages(where="src")
META_PATH = os.path.join("src", "jaxstar", "__init__.py")
CLASSIFIERS = [
    "Programming Language :: Python",
]
INSTALL_REQUIRES = [
    "arviz>=0.15",
    "astropy>=5",
    "jax>=0.4.30",
    "jaxlib>=0.4.30",
    "numpy>=1.23",
    "numpyro>=0.15",
    "pandas>=1.5",
    "scipy>=1.9",
]
TEST_REQUIRES = ["pytest>=7"]

# END PROJECT SPECIFIC
HERE = os.path.dirname(os.path.realpath(__file__))


def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


def find_meta(meta, meta_file=read(META_PATH)):
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), meta_file, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


if __name__ == "__main__":
    setup(
        name=NAME,
        version="0.1.0",
        author=find_meta("author"),
        author_email=find_meta("email"),
        maintainer=find_meta("author"),
        maintainer_email=find_meta("email"),
        url=find_meta("uri"),
        license=find_meta("license"),
        description=find_meta("description"),
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        packages=PACKAGES,
        package_dir={"": "src"},
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        extras_require={"test": TEST_REQUIRES},
        python_requires=">=3.10",
        classifiers=CLASSIFIERS,
        zip_safe=False,
    )
