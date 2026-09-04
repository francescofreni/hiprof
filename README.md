![](https://img.shields.io/badge/python-≥3.11-blue)
[![PyPI version](https://img.shields.io/pypi/v/hiprof.svg?color=orange&logo=pypi)](https://pypi.org/project/hiprof/)
[![read](https://badgen.net/badge/read/arXiv/b31b1b)](https://arxiv.org/abs/2607.13883)

# Hi Prof 👋: High-Probability Falsifier

> We formalize _verification_ in causal graphical models: 
> deciding whether a given observational formula identifies 
> a target interventional distribution. This opens a problem 
> complementary to identification, asking not whether any 
> identifying formula exists, but whether the given formula is
> identifying. We show that even sound and complete solutions to
> identification do not solve verification. We propose a 
> falsifier as a first practical route forward, prove that it 
> induces an almost-surely correct verifier for regular 
> exponential-family models, and describe the gateway test,
> a natural application of the resulting verifier that 
> finds all sets admissible for use in a front-door formula.

If you publish research using `hiprof`, please cite
[our paper](https://arxiv.org/pdf/2607.13883) introducing verification:
```bibtex
@article{hiprof2026,
  author  = {Francesco Freni and Leonard Henckel and Sebastian Weichwald},
  title   = {{Verifying formulas for interventional distributions}},
  journal = {{arXiv preprint arXiv:2607.13883}},
  year    = {2026}
}
```

[Feedback](https://github.com/francescofreni/hiprof/issues/new/choose) very welcome!

## ⚙️ Installation

Install `hiprof` from PyPI using `pip`:

```bash
# create virtual environment (optional)
python -m venv venv_hiprof
source venv_hiprof/bin/activate

# install hiprof
python -m pip install hiprof
```

Alternatively, using `uv`:

```bash
# create virtual environment (optional)
uv venv

# install hiprof
uv pip install hiprof
```

Two optional feature gates provide additional functionality:

- `identification` installs
  [`y0`](https://github.com/y0-causal-inference/y0) and enables
  `IDAlgorithm`, which derives an identifying formula when one is available;
- `nonidentifiability` installs
  [`ananke-causal`](https://gitlab.com/causal/ananke) and enables verification
  of non-identifiability claims.

Install optional features individually:

```bash
python -m pip install "hiprof[identification]"
python -m pip install "hiprof[nonidentifiability]"
```

or install all optional features:

```bash
python -m pip install "hiprof[all]"
```

The same requirement strings can be used with `uv pip install`.


## 🚀 Quick start

```python
from hiprof import HPFalsifier

falsifier = HPFalsifier(
    graph="T -> M; M -> Y; T <-> Y",
    treatments="T",
    outcomes="Y",
)

formula = """
sum_{M} {
    p(M | T)
    sum_{T'} { p(Y | M, T') p(T') }
}
"""

falsifier.check(formula)
# Output:
# True
# False-acceptance bound: 5.421e-18
```

Adjacency matrices can be translated as follows (`numpy` arrays and `pandas` DataFrames are accepted as well):

```python
from hiprof import adjacency_to_graph

adjacency = [
    [0, 1, 1],
    [0, 0, 1],
    [1, 0, 0],
]

graph = adjacency_to_graph(
    adjacency,
    nodes=["T", "M", "Y"],
    edge_direction="from row to column",
)
# T -> M
# T <-> Y
# M -> Y
```

## 📚 Examples

For a concise introduction, see
[`notebooks/demo.ipynb`](https://github.com/francescofreni/hiprof/blob/main/notebooks/demo.ipynb).

After cloning and installing the repository, the standalone examples in
[`examples/`](https://github.com/francescofreni/hiprof/tree/main/examples/)
can be run from the repository root:

```bash
python examples/frontdoor.py
python examples/napkin.py
python examples/cross_graph_verification.py
```

`cross_graph_verification.py` uses `IDAlgorithm` and therefore requires the
`identification` feature.


## 🛠️ Local development

Clone the repository and install it in editable mode:

```bash
git clone git@github.com:francescofreni/hiprof.git
cd hiprof
python -m pip install -e ".[dev]"
```

To include all optional features as well as the development tools:

```bash
python -m pip install -e ".[dev,all]"
```

Run the test suite with
```bash
pytest
```
