# scikit-pns

[![Supported Python Versions](https://img.shields.io/pypi/pyversions/scikit-pns.svg)](https://pypi.python.org/pypi/scikit-pns/)
[![PyPI Version](https://img.shields.io/pypi/v/scikit-pns.svg)](https://pypi.python.org/pypi/scikit-pns/)
[![License](https://img.shields.io/github/license/JSS95/scikit-pns)](https://github.com/JSS95/scikit-pns/blob/master/LICENSE)
[![CI](https://github.com/JSS95/scikit-pns/actions/workflows/ci.yml/badge.svg)](https://github.com/JSS95/scikit-pns/actions/workflows/ci.yml)
[![CD](https://github.com/JSS95/scikit-pns/actions/workflows/cd.yml/badge.svg)](https://github.com/JSS95/scikit-pns/actions/workflows/cd.yml)
[![Docs](https://readthedocs.org/projects/scikit-pns/badge/?version=latest)](https://scikit-pns.readthedocs.io/en/latest/?badge=latest)

![title](https://scikit-pns.readthedocs.io/en/latest/_images/plot-header.png)

Principal nested spheres analysis for scikit-learn.

## Usage

```python
>>> from skpns import IntrinsicPNS
>>> from skpns.util import circular_data
>>> X = circular_data()
>>> X_new = IntrinsicPNS().fit_transform(X)
```

## Installation

```
$ pip install scikit-pns
```

## Documentation

The manual can be found online:

> https://scikit-pns.readthedocs.io

If you want to build the documentation yourself, get the source code and install with `[doc]` dependency.
Then, go to the `doc` directory and build the documentation:

```
$ pip install .[doc]
$ cd doc
$ make html
```

The documentation will be generated in the `build/html` directory. Open `index.html` to see the main page.

## Developing

### Installation

For development features, you must install the package by `pip install -e .[dev]`.
