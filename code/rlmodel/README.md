# Overview 

This document summarizes the design decisions and implementation rationale
underlying the models used in this work.

# Models Scope

**Objective:**
- Identify a model that explains the substantial variability in sampling time.
- Favor implementations that are straightforward and interpretable.
    - Each model variant reflects a distinct behavioral phenotype.
        - Q-learning captures choice bias; R-learning captures reward rate.
    - Use a minimal and consistent parameterization.
        - No separate parameters for rewarded vs. non-rewarded outcomes.
        - No separate learning rates for left vs. right states.
        - Non-visited Q-states, i.e. *$Q^n_L$* and *$Q^n_R$*, are not updated
          (i.e., they do not decay toward a baseline value).
    - Use *textbook* formulations.
        - Exception: Q-learning includes an additional parameter, *$offset$*.
        - *$offset$* represents a subject-specific, constant motor or cognitive
          bias.

**What it isn't:**
- An attempt to maximize goodness of fit.
   - Several previous studies (e.g., , , , ) have achieved strong model fits.
   - Many prior works use log-likelihood–based fitting, which can improve fit
     quality at the cost of increased computational complexity. We are open to
     adopting this approach if reviewers deem it more appropriate.
- A claim of novelty in combining Q-learning or R-learning individually with
  DDM.
    - Previous work on DDM Q-learning:
    - Previous work on DDM R-learning:
    - **Exception**: To the best of our knowledge, this is the first work to
      combine both approaches within a single framework.
- An effort to reduce the number of parameters to the absolute minimum.
    - For example, in the combined Q- and R-learning model, *$RR^n$* could be
      inferred from *$Q^n_L$* and *$Q^n_R$*.
        - This would complicate direct model comparison.
        - This approach may not generalize to settings with more than two
          choices beyond *$Q_L$* and *$Q_R$*.
- An exploration of alternative DDM variants.
  - Extensions such as collapsing decision boundaries [ref](), or step-wise
    drift-rate modulation [ref](), are not considered.
  - Although such variants may yield improved fits, our interpretation does not
    justify favoring one formulation over another. We therefore adopt the
    standard DDM.
  - We are open to including alternative variants if reviewers believe they
    would improve interpretability.

<ins>What is the value of this work?</ins>

- Demonstrates limitations in capturing behavioral phenomena using a single RL
  model alone, or without RL-based modeling.
- Given the strong influence of trial history reported in our paper—specifically
  reward rate and bias—these models provide a principled framework for capturing
  these behavioral phenotypes.
- The adapted formulations are intended as generalizations that future work can
  tailor to specific experimental contexts and findings.

# Implementation Details

TODO: Package structure and dependencies  
TODO: Exclusion criteria: subjects with fewer than 2,500 trials are excluded  
TODO: Preprocessing steps applied to enable vectorized computation  

# Fit results

TODO: Link to the results directory  
TODO: Description of reported metrics and figures  

# Schematic-like figure (Fig. 5f, middle)

![Schematic-like figure](/results/model_schematic/Q_R_Heatmap.svg)

TODO: Include representative example image  
TODO: Reference the notebook and simulations based on fitted parameters 