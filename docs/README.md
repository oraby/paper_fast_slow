# `docs/` — manuscript ↔ repository reference

Reference material tying this repository to the manuscript it supports:

> **Nashaat, Oraby, Krasniqi, Goh-Sauerbier, Bosc, Koerner, Lobova, Karayel,
> Kepecs, Larkum. "The neural mechanisms of fast versus slow decision-making."**
> *Neuron* (2026, advance online publication). [doi:10.1016/j.neuron.2026.10.024](https://doi.org/10.1016/j.neuron.2026.10.024)

These documents tie the paper to the code: where each figure panel comes
from, and which module implements each analysis in the Methods.

| Document | What it is for |
|---|---|
| [`manuscript-figure-map.md`](manuscript-figure-map.md) | Panel-by-panel map: every main and supplementary figure panel → the notebook, module and `results/` artifact that produces it. Includes the reported numbers and statistics per panel. |
| [`manuscript-methods-map.md`](manuscript-methods-map.md) | Methods section → code. Every analysis described in *Quantification and Statistical Analysis*, including all model equations as written in the paper, with the module that implements it. |
| [`model-fitting.md`](model-fitting.md) | Design rationale and validation of the RL-DDM fitting (χ², MLE and the joint criterion), kept for reference; the code and `code/rlmodel/README.md` are authoritative. |

## Conventions used in these documents

- **Figure numbering is the published numbering** (`Figure 1`–`Figure 7`,
  `Figure S1`–`Figure S14`), in these documents and in the notebook section
  headings alike.
- **"Not from this repo"** means the panel is a schematic, 3D render, photograph
  or micrograph produced outside the analysis code. The manuscript states that
  some panels are not generated here; those are called out explicitly rather
  than left as gaps.
- **✗ not drawn** marks a published panel that no run of the shipped notebooks
  reproduces, with the reason (no cell draws it, or the example drawn from a
  per-neuron / per-subject loop was never recorded).
