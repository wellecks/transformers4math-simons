# Transformers for Mathematics Tutorial

Simons Institute and SLMath Joint Workshop: AI for Mathematics and Theoretical Computer Science, April 8 2025


### Contents
We provide Google Colab notebooks that have the required environment and files.

1. **Bigram model**: <a href="https://colab.research.google.com/github/wellecks/transformers4math-simons/blob/main/1_bigram/bigrams_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
      - A very simple language model based on counting consecutive tokens.

2. **Transformer**: <a href="https://colab.research.google.com/github/wellecks/transformers4math-simons/blob/main/2_transformer/transformer_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
      - Implement and train a simple Transformer language model.

3. **Addition**: <a href="https://colab.research.google.com/github/wellecks/transformers4math-simons/blob/main/3_addition/addition_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
      - Train a model for four-digit addition.
4. **Triangle-free graphs**: <a href="https://colab.research.google.com/github/wellecks/transformers4math-simons/blob/main/4_graphs/graphs_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
      - Train a model to generate triangle-free graphs.

*Note*: you can safely skip notebook 1 and 2 if you are primarily interested in using existing transformer libraries for different applications.

## (Optional) Local Setup

If you would like to run the notebooks locally instead of on Colab please follow these instructions.

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

