# Contributing

Any contributions to NAV123D are welcome! Complete guidelines will be added along the way.

## Setting up a development environment

We recommend a fresh conda environment with Python 3.12 (any of Python 3.9–3.12 works).

1. Clone the repository:

   ```bash
   git clone git@github.com:DanielDauner/nav123d.git
   cd nav123d
   ```

2. Create and activate the environment:

   ```bash
   conda create -n nav123d python=3.12 -y
   conda activate nav123d
   ```

3. Install the package with its development dependencies (editable install):

   ```bash
   pip install uv
   uv pip install -e .[dev]
   ```

   The `dev` extra adds `pyright`, `ruff`, `pre-commit`, `pytest`, and `pytest-cov`.
   To also build the documentation locally, install the `docs` extra
   (`uv pip install -e .[dev,docs]`).

4. Install the pre-commit hooks so they run automatically on every commit:

   ```bash
   pre-commit install
   ```

   To check that everything is wired up correctly, run the hooks against the
   whole codebase once:

   ```bash
   pre-commit run --all-files
   ```

## Before opening a pull request

- Run the linters and formatter (these also run via pre-commit): `ruff check .` and `ruff format .`.
- Run the test suite: `pytest`. (kinda useless atm).
