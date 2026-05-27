# MLE Population Notebook TODO

## Summary
Create a notebook-driven MLE exploration tool under `rlmodel/model/mle_notebooks/`, with most logic in reusable Python modules and notebook cells acting only as configuration and display calls.

## Tasks
- Add `mle_population_explorer.ipynb` with the existing relative-import first-cell trick, adjusted to `root_parent_level = 3`.
- Add reusable backend modules for:
  - loading MLE and matching posterior predictive result pickles from `../data/RLModel`;
  - linked latent histograms with click/query/undo filtering;
  - fitted-result selector and parameter slider specs;
  - single-trial DDM frame generation, plotting, and TIFF export.
- Keep notebook cells minimal: load data, show latent histogram explorer, show DDM viewer.
- Add unit tests for loading, filtering, slider specs, DDM frame generation, and notebook JSON smoke validation.

## Acceptance Checks
- Histogram filters are stack-based, undoable, and non-mutating.
- Histogram bin clicks use numeric bin ranges.
- DDM viewer uses existing MLE latent/first-passage functions rather than reimplementing MLE math.
- Save stack writes a multi-page TIFF when PIL is installed.
- Existing dirty notebooks and cache files are not modified.
