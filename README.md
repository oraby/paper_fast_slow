

This repository contains the supporting materials for the paper **The neural
mechanisms of fast versus slow decision-making**.

# Repository structure

- [`data/`](data/)
  Data used to generate the results reported in the paper.

- [`code/`](code/)
  Jupyter notebooks for data analysis and figure generation.

- [`results/`](results/)
  Generated figures, including per-subject results.

# Notes

- The [`DDM_RL-Model`](code/rlmodel/) package README provides additional details
  on model design and implementation choices.
- The [`Code/README.md`](code/) file describes the data organization and
  expected directory structure.
- Some two-photon imaging datasets are too large to be included directly in the
  repository. Running  [`code/data_downloader.ipynb`](code/data_downloader.ipynb)
  will download the missing data.
