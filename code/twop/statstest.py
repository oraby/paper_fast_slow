import warnings

import numpy as np
import scipy.stats as stats

#: Shapiro-Wilk (normality) and Levene (equal-variance) decision threshold.
NORMALITY_ALPHA = 0.05

TEST_MWU = "MannWhitneyU"
TEST_TTEST_STUDENT = "t-test (Student)"
TEST_TTEST_WELCH = "t-test (Welch)"


def pStars(pval, na="n.a."):
  """``***`` / ``**`` / ``*`` / ``n.s.`` for a p-value (``na`` when unavailable)."""
  pval = np.nan if pval is None else float(pval)
  if np.isnan(pval):
    return na
  return ("***" if pval <= 0.001 else "**" if pval <= 0.01 else
          "*" if pval <= 0.05 else "n.s.")


def shapiroNormality(vals, alpha=NORMALITY_ALPHA):
  """Shapiro-Wilk on one sample -> ``(W, pval, is_normal)``.

  NaN (and "not normal") when the sample is too small for the test or is
  constant -- both leave normality unestablished, so a caller gating on this
  should fall back to the non-parametric test.
  """
  vals = np.asarray(vals, dtype=float)
  vals = vals[~np.isnan(vals)]
  if len(vals) < 3:  # Shapiro-Wilk needs at least 3 observations
    return np.nan, np.nan, False
  try:
    with warnings.catch_warnings():  # constant input -> nan, not a crash
      warnings.simplefilter("ignore", RuntimeWarning)
      W, pval = stats.shapiro(vals)
  except ValueError:  # e.g. constant input
    return np.nan, np.nan, False
  if np.isnan(pval):
    return W, pval, False
  return W, pval, bool(pval > alpha)


def normalityGatedTest(left_vals, right_vals, alpha=NORMALITY_ALPHA,
                       left_name="left", right_name="right"):
  """Two independent samples, with the test chosen by a normality check.

  Runs Shapiro-Wilk on *each* sample first (they are independent samples, so
  there are no paired differences to test). Both normal -> two-sample t-test,
  with Levene's test picking Student (equal variances) vs Welch; otherwise
  Mann-Whitney U. All two-sided.

  Returns a dict with ``{left_name,right_name}_shapiro_W/_shapiro_pval``,
  ``is_normal``, ``levene_pval``, then ``test``/``statistic``/``pval`` for the
  test the gate selected, plus ``ttest_pval`` and ``mwu_pval`` -- both are
  always computed when possible, so the choice can be second-guessed.
  """
  left = np.asarray(left_vals, dtype=float)
  right = np.asarray(right_vals, dtype=float)
  left, right = left[~np.isnan(left)], right[~np.isnan(right)]

  row = {f"{left_name}_shapiro_W": np.nan, f"{left_name}_shapiro_pval": np.nan,
         f"{right_name}_shapiro_W": np.nan,
         f"{right_name}_shapiro_pval": np.nan,
         "is_normal": False, "levene_pval": np.nan, "test": TEST_MWU,
         "statistic": np.nan, "pval": np.nan,
         "ttest_pval": np.nan, "mwu_pval": np.nan}
  if len(left) < 1 or len(right) < 1:
    return row

  both_normal = True
  for name, vals in [(left_name, left), (right_name, right)]:
    W, pval, is_normal = shapiroNormality(vals, alpha)
    row[f"{name}_shapiro_W"] = W
    row[f"{name}_shapiro_pval"] = pval
    both_normal = both_normal and is_normal
  row["is_normal"] = both_normal

  mwu = StatsTest.mannwhitneyu(left, right).resultDict()
  row["mwu_pval"] = mwu["pval"]

  t_res = None
  if len(left) >= 2 and len(right) >= 2:
    try:
      with warnings.catch_warnings():  # degenerate (constant) samples
        warnings.simplefilter("ignore", RuntimeWarning)
        row["levene_pval"] = stats.levene(left, right).pvalue
        t_res = stats.ttest_ind(left, right,
                                equal_var=row["levene_pval"] > alpha)
      row["ttest_pval"] = t_res.pvalue
    except ValueError:  # e.g. constant input to Levene
      t_res = None

  if row["is_normal"] and t_res is not None:
    row["test"] = (TEST_TTEST_STUDENT if row["levene_pval"] > alpha
                   else TEST_TTEST_WELCH)
    row["statistic"], row["pval"] = t_res.statistic, t_res.pvalue
  else:
    row["test"] = TEST_MWU
    row["statistic"], row["pval"] = mwu["statistic"], mwu["pval"]
  return row


class StatsTest:
  class TestResult:
    def __init__(self, results_dict):
      self._results_dict = results_dict

    def resultDict(self):
      return self._results_dict

  @staticmethod
  def mannwhitneyu(left_data, right_data):
    if isinstance(left_data, str) and left_data == "name":
      return "MannWhitneyU"
    test_result = stats.mannwhitneyu(left_data, right_data)
    return StatsTest.TestResult({"pval":test_result.pvalue,
                                "statistic":test_result.statistic,
                                "num_samples_left":len(left_data),
                                "num_samples_right":len(right_data)})

  @staticmethod
  def AUCPermutations(left_data, right_data, track_xs_ys=False,
                      skip_permutations=False, permutation_test_kargs={}):
    if isinstance(left_data, str) and left_data == "name":
      return "AUC_Permutations"
    all_data = np.concatenate((left_data, right_data))
    min_val = np.min(all_data) - 1
    all_data_sorted = np.sort(np.concatenate((all_data, [min_val])))
    auc = _AUC(all_data_sorted, track_xs_ys=track_xs_ys)
    if skip_permutations:
      pval = np.nan
      statistic = auc.calcArea(left_data, right_data)
    else:
      test_result = stats.permutation_test((left_data, right_data), auc.calcArea,
                                         vectorized=True,
                                         **permutation_test_kargs)
      pval = test_result.pvalue
      statistic = test_result.statistic
    return StatsTest.TestResult({"pval":pval,
                                 "statistic":statistic,
                                 "num_samples_left":len(left_data),
                                 "num_samples_right":len(right_data)})


class _AUC:
  def __init__(self, all_data_sorted, track_xs_ys=False):
    self._all_data_sorted = all_data_sorted
    self._track_xs_ys = track_xs_ys
    if self._track_xs_ys:
      self.xs = []
      self.ys = []

  def calcArea(self, first_data, second_data, axis=-1):
    if first_data.ndim == 1:
      shape = len(self._all_data_sorted)
      two_dim = False
    elif first_data.ndim == 2:
      shape = [first_data.shape[0], len(self._all_data_sorted)]
      two_dim = True
    else:
      raise ValueError("easy_data.ndim:", first_data.ndim)
    len_first = first_data.shape[-1]
    len_second = second_data.shape[-1]
    prob_x = np.zeros(shape)
    prob_y = np.zeros(shape)
    for idx, data_pt in np.ndenumerate(self._all_data_sorted):
      if two_dim:
        prob_x[:,idx] = \
                   np.sum(first_data <= data_pt, axis=1)[:,np.newaxis]/len_first
        prob_y[:,idx] = \
                 np.sum(second_data <= data_pt, axis=1)[:,np.newaxis]/len_second
      else:
        prob_x[idx] = np.sum(first_data <= data_pt)/len_first
        prob_y[idx] = np.sum(second_data <= data_pt)/len_second
    trapz_area = np.trapz(prob_y, prob_x, axis=axis)
    # print("trapz_area:", trapz_area)
    # simpson_area = simpson(np.array(prob_y), np.array(prob_x))
    if self._track_xs_ys:
      if two_dim:
        self.xs += list(prob_x)
        self.ys += list(prob_y)
      else:
        self.xs.append(prob_x)
        self.ys.append(prob_y)
    return trapz_area

  def plot(self, ax=None):
    assert self._track_xs_ys
    if ax is None:
      ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    xs, ys = np.array(self.xs).squeeze(), np.array(self.ys).squeeze()
    print("auc.xs:", xs.shape, "auc.ys:", ys.shape)
    ax.plot(xs, ys, color="k")
    ax.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), color="grey",
            ls="--", alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax
