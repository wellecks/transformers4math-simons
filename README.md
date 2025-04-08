# Transformers for Mathematics Tutorial

Simons Institute and SLMath Joint Workshop: AI for Mathematics and Theoretical Computer Science, April 8 2025


### Contents
#### Background and implementation
Work through these notebooks to see details of how a transformer language model is implemented and how training is implemented.

1. **Notebook 1: Bigram model**: a very simple language model based on counting consecutive tokens.
2. **Notebook 2: Transformer**: implement a train a simple Transformer language model.

*Note*: you can safely skip notebook 1 and 2 if you are primarily interested in using existing transformer libraries for different applications.

#### Applications:
Work through these notebooks to see how to use the Makemore library to train transformers on different datasets.

3. **Notebook 3: Addition**: train a model for four-digit addition.
4. **Notebook 4: Triangle-free graphs**: train a model to generate triangle-free graphs.

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

