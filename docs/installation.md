# Installation

## Pip-Install

```bash
mkdir -p $HOME/nav123d_workspace; cd $HOME/nav123d_workspace # Optional
git clone git@github.com:DanielDauner/nav123d.git
cd nav123d
pip install -e .
```

## Dataset set-up
See [123D docs](https://kesai.eu/py123d/).

## Environment variables

```bash
export PY123D_DATA_ROOT=...
export NAV123D_EXP_ROOT="$HOME/nav123d_workspace/exp"
```
