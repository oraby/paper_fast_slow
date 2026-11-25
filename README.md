# The neural mechanisms of fast versus slow decision-making

Code, data and figures for

> Nashaat, M., Oraby, H., Krasniqi, F., Goh-Sauerbier, S. T., Bosc, M.,
> Koerner, S., Lobova, A., Karayel, S., Kepecs, A., & Larkum, M. E. (2026).
> The neural mechanisms of fast versus slow decision-making. *Neuron*.
> Advance online publication.
> https://doi.org/10.1016/j.neuron.2026.10.024

The notebooks in [`code/`](code/) reproduce the analyses and figure panels of
the paper from the data in [`data/`](data/); the figures they write are in
[`results/`](results/).

## Repository structure

- [`code/`](code/) — Jupyter notebooks, one per experiment type, calling tested
  Python modules. [`code/README.md`](code/README.md) lists the notebooks, the
  data format and how to run them.
- [`code/rlmodel/`](code/rlmodel/) — the RL-enhanced drift-diffusion model;
  [`code/rlmodel/README.md`](code/rlmodel/README.md) states the model and its
  equations.
- [`data/`](data/) — the data behind every figure. The largest two-photon
  files are not in the repository; [`code/data_downloader.ipynb`](code/data_downloader.ipynb)
  fetches them from Zenodo, where the complete `data/` folder is also archived:
  [doi:10.5281/zenodo.24260552](https://doi.org/10.5281/zenodo.24260552).
  The code is archived at
  [doi:10.5281/zenodo.24496982](https://doi.org/10.5281/zenodo.24496982), which
  always points to its latest version.
- [`results/`](results/) — the figures the notebooks write, including
  per-subject panels.
- [`docs/`](docs/) — [`manuscript-figure-map.md`](docs/manuscript-figure-map.md)
  maps every figure panel to the notebook section, module and output file that
  produce it; [`manuscript-methods-map.md`](docs/manuscript-methods-map.md) maps
  the Methods to the code; [`model-fitting.md`](docs/model-fitting.md) records
  how the model fitting was designed and validated.

## Getting started

The environment is managed by [`uv`](https://docs.astral.sh/uv/) from
`pyproject.toml` and `uv.lock`:

```
uv sync                                          # create .venv from the lockfile
uv run pytest                                    # run the test suite
uv run jupyter lab                               # open the notebooks
uv run python code/run_notebooks.py --list       # what the runner would execute
uv run python code/run_notebooks.py --save-figs  # run every figure notebook, writing results/
```

Run `code/data_downloader.ipynb` once before the two-photon notebooks. Each
notebook section is headed by the figure panel it draws (e.g. "Figure 4H"), so
a panel can be found by searching the notebooks for its number.

## Third-party code

- [`code/figcode/psychofit/`](code/figcode/psychofit/) is
  [psychofit](https://github.com/cortex-lab/psychofit) by the International
  Brain Laboratory, under the MIT licence in that folder: upstream commit
  `7335497` (July 2024), with one change, an optional `ncpus` that lets
  `mle_fit_psycho` run its starting points in parallel.
- [`code/twop/relogit/`](code/twop/relogit/) is relogit by Arman
  Hassanniakalager, a rare-events logistic regression (King and Zeng, 2001),
  under the MIT licence in that folder.
- The ΔF/F baseline in `code/pipeline/tracesnormalize.py` uses the maximin
  method of suite2p (Pachitariu et al., 2017); no suite2p code is included.
- A few short snippets come from Stack Overflow answers (CC BY-SA 4.0); each is
  marked in the code with a link to the answer it comes from.

## License

The code is released under the [MIT License](LICENSE). The data in
[`data/`](data/) and the figures in [`results/`](results/) are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The bundled
third-party code keeps its own licence (see above). If you use the code or the
data, please cite the paper; [`CITATION.cff`](CITATION.cff) has the details.
