import numpy as np
import scipy.stats as stats



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
