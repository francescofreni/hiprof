![](https://img.shields.io/badge/python-≥3.10-blue)

# Hi Prof 👋: High-Probability Falsifier

Early access 🐣 [Feedback](https://github.com/francescofreni/hiprof/issues/new/choose) very welcome!

## ⚙️ Installation

`hiprof` can be installed directly from GitHub using `pip`:
``` zsh
# create virtual environment (optional)
python -m venv venv_hiprof
source venv_hpv/bin/activate

# install hiprof
pip install "git+https://github.com/francescofreni/hiprof.git"
```

Alternatively, using `uv`:
``` zsh
# create virtual environment (optional)
uv venv

# install hiprof
uv pip install "git+https://github.com/francescofreni/hiprof.git"
```

For local development, clone the repository and install it in editable mode:
``` zsh
git clone git@github.com:francescofreni/hiprof.git
cd hiprof
pip install -e .
```

## 🚀 Usage

The code in [notebooks/](./notebooks/) demonstrates the core functionalities and main functions of `hiprof`.
For an introduction, see [notebooks/demo.ipynb](./notebooks/demo.ipynb).

## 📜 Citation

If you use `hiprof` in your scientific work, please cite [this paper](https://arxiv.org/abs/2607.13883) introducing verification:
```bibtex
@article{hiprof2026,
  author  = {Francesco Freni and Leonard Henckel and Sebastian Weichwald},
  title   = {{Verifying formulas for interventional distributions}},
  journal = {{arXiv preprint arXiv:2607.13883}},
  year    = {2026}
}
```