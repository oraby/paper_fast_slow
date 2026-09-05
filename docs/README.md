# `docs/` — manuscript ↔ repository reference

Reference material tying this repository to the manuscript it supports:

> **Nashaat, Oraby, Krasniqi, Goh-Sauerbier, Bosc, Koerner, Lobova, Karayel, …
> Larkum. "Cortical mechanisms of fast versus slow decision making."**
> *Neuron*, revision draft (`Nashaat et al.,_Neuron_Revision.docx`).

The manuscript is the reference; this repository is the implementation. These
documents exist so that any cleanup, refactor or deletion can be checked
against "does the paper still build?".

| Document | What it is for |
|---|---|
| [`manuscript-figure-map.md`](manuscript-figure-map.md) | Panel-by-panel map: every main and supplementary figure panel → the notebook, module and `results/` artifact that produces it. Includes the reported numbers and statistics per panel. |
| [`manuscript-methods-map.md`](manuscript-methods-map.md) | Methods section → code. Every analysis described in *Quantification and Statistical Analysis*, including all model equations as written in the paper, with the module that implements it. |
| [`repo-audit.md`](repo-audit.md) | Findings from the mapping pass: dead code, orphaned backends, environment fragmentation, test gaps, stale documentation. This is the input for the cleanup work. |
| [`data-portability.md`](data-portability.md) | Why most artifacts under `data/` load in only one interpreter, and the one-off migration that removes the dependency. Read before touching `data/`. |
| [`execution-plan.md`](execution-plan.md) | Dependency analysis of the cleanup workstreams: what blocks what, which forks can run concurrently without colliding, and the six decisions that gate the work. |

## Conventions used in these documents

- **Figure numbering is the manuscript's current numbering** (`Figure 1`–`Figure 7`,
  `Figure S1`–`Figure S14`).
- Notebook section headings still carry **older numbering** from previous
  submissions and are *not* reliable. Where a notebook heading is quoted it is
  marked as such. See [the numbering-drift note](#numbering-drift) below.
- **"Not from this repo"** means the panel is a schematic, 3D render, photograph
  or micrograph produced outside the analysis code. The manuscript states that
  some panels are not generated here; those are called out explicitly rather
  than left as gaps.
- Confidence is flagged only where it is less than certain:
  - *(inferred)* — matched by panel content + output filename, no explicit label
    in the code.

## Numbering drift

Three generations of figure numbering coexist in the notebooks. When reading a
notebook heading, translate it:

| Notebook | Numbering generation | Translation to current |
|---|---|---|
| `behavior.ipynb` | Current-ish | `Fig. N` → `Figure N`; `Ext. Fig. N` → `Figure SN`. Panel **letters** within `Figure S2`/`S3` have shifted — trust content, not the letter. |
| `opto.ipynb`, `widefield.ipynb`, `TwoPTraces.ipynb` | One main figure behind | `Fig. N` → `Figure N+1`; `Extended Fig. N` → `Figure S(N−1)`. |
| `2pAnalysis.ipynb` | Mixed | `Fig. 5F` → `Figure 5F` (current); `Fig. S8x` → `Figure S12x`. |
| `Tracking.ipynb` | Own scheme | `Extended Fig. 4b–e` → `Figure S3J–M`. |
| `rlmodel/*.ipynb`, `rlmodel/README.md` | Oldest | `Fig. 1k` → `Figure 2E`; `Fig. 1l` → `Figure 2G`; `Fig. 5f middle` → `Figure 7D`; `Ext. Fig. 5a` → `Figure S4A/C`. |

Renumbering the headings to the current scheme is listed as a cleanup task in
[`repo-audit.md`](repo-audit.md).
