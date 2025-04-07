# Transformers for Math Tutorial

## Setup

Install [uv](https://github.com/astral-sh/uv). See [installation
instructions](https://docs.astral.sh/uv/getting-started/installation/)

Execute, in the terminal
```
uv sync
uv venv
```

These two commands download dependencies and create a virtual environment. The
second command shows the instruction for activating the virtual environment.
After activating the venv, check that Torch is available by

```
python3 -c "import torch"
```

To launch Jupyter notebook, run inside the virtual environment

``` c
jupyter notebook
```

