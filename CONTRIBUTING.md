# Contributing

Use a focused branch and keep generated meshes out of commits unless they are
small review fixtures. Before opening a pull request, run:

```sh
python -m pip install -e ".[dev]"
python -m ruff check src examples tests
python -m ruff format --check src examples tests
python -m pytest
python -m build
python -m twine check dist/*
```

Changes to mesh generation should include a regression test and report their
effect on invalid cells and the compact quality metrics. Use the NACA0012
example as a common comparison case, but add a smaller focused fixture when it
can expose the behavior more directly.

Do not add third-party airfoil collections, proprietary data, credentials, or
large generated meshes. Any code copied or adapted from another project must
retain all required license and copyright notices in the same pull request.
