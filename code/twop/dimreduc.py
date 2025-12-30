from ..pipeline import pipeline
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

class DimReduc:
  @staticmethod
  def mean(data):
    if isinstance(data, str) and data == "name":
      return "mean"
    try:
      data = np.array(data)
    except ValueError: # Deal with non-equal length data
      data_mean = np.array([np.nanmean(data_pt) for data_pt in data])
    else:
      data_mean = np.nanmean(data, axis=1)
    return data_mean, {}

  @staticmethod
  def max(data):
    if isinstance(data, str) and data == "name":
      return "max"
    try:
      data = np.array(data)
    except ValueError: # Deal with non-equal length data
      data_max = np.array([np.nanmax(data_pt) for data_pt in data])
    else:
      data_max = np.nanmax(data, axis=1)
    return data_max, {}

  _pca = PCA(n_components=1)
  @staticmethod
  def pca(data, plot_trace_id=None):
    if isinstance(data, str) and data == "name":
      return "PCA_PC1"
    data = np.array(data)
    data = StandardScaler().fit_transform(data)
    pca_features = DimReduc._pca.fit_transform(data).squeeze()
    assert pca_features.ndim == 1
    assert len(pca_features) == data.shape[0]
    return pca_features, {
      "explained_variance_ratio":DimReduc._pca.explained_variance_ratio_[0],
    }

  _n_neighbors = 8
  # _isomap = Isomap(n_components=1, n_neighbors=_n_neighbors)
  @staticmethod
  def isomap(data):
    if isinstance(data, str) and data == "name":
      return f"ISOMAP_embd1_k{DimReduc._n_neighbors}"
    data = np.array(data)
    data = StandardScaler().fit_transform(data)
    # isomapFn = DimReduc._isomap
    isomapFn = Isomap(n_components=1, n_neighbors=DimReduc._n_neighbors)
    isomap_features = isomapFn.fit_transform(data).squeeze()
    assert isomap_features.ndim == 1
    assert len(isomap_features) == data.shape[0]
    #print("Data shape:", data.shape, "- ismo_features:", isomap_features.shape)
    reconstruction_err = isomapFn.reconstruction_error()
    return isomap_features, {
      "reconstruction_err":reconstruction_err,
    }

class DimReducWrapper(pipeline.DFProcessor):
  '''
  dim_reduc_wrapper = DimReducWrapper(dimReducFunc=dimReducFn,
                                      set_name="neuronal")
  dim_redic_name = dim_reduc_wrapper.dimReducName()
  print("Running dim reduc:", dim_redic_name)
  chain =  pipeline.Chain(
      pipeline.BySession(),
        pipeline.LoopTraces(dim_reduc_wrapper),
      pipeline.RecombineResults(),
  )
  df = chain.run(df)
  new_rows = []
  for sess, sess_df in df.groupby("ShortName"):
    sess_trial_traces = dim_reduc_wrapper._sess_trials_traces_res[sess]
    for row_idx, row in sess_df.iterrows():
      traces_sets = pipeline.getRowTracesSets(row)
      traces_neuronal = traces_sets["neuronal"]
      dim_reduc_neuronal = sess_trial_traces[row.TrialNumber]
      assert traces_neuronal.keys() == dim_reduc_neuronal.keys()
      row = row.copy()
      row["neuronal_reduc"] = dim_reduc_neuronal
      new_rows.append(row)
  df = pd.DataFrame(new_rows)
  '''
  def __init__(self, set_name, dimReducFn):
    self._set_name = set_name
    self._dimReducFn = dimReducFn
    self._sess_trials_traces_res = {}
    self._sess_trace_reduc_info = {}

  def process(self, df):
    trial_traces = {}
    trace_id = None
    sess_name = df.ShortName.unique()
    assert len(sess_name) == 1
    sess_name = sess_name[0]
    for row_idx, row in df.iterrows():
      traces_sets = pipeline.getRowTracesSets(row)
      for traces_set_name, traces_set in traces_sets.items():
        if traces_set_name != self._set_name:
          continue
        assert len(traces_set) == 1, "Didn't handle yet multiple traces"
        cur_trace_id = list(traces_set.keys())[0]
        if trace_id is None:
          trace_id = cur_trace_id
        else:
          assert trace_id == cur_trace_id, "Didn't handle yet multiple traces"
        # TODO: Add axis here
        trace = traces_set[trace_id].take(range(row.trace_start_idx,
                                                row.trace_end_idx+1))
        trial_traces[row.TrialNumber] = trace
    assert len(trial_traces), "Did you pass the wrong set_name?"
    # print("Traces shape:", traces.shape)
    traces = list(trial_traces.values())
    dim_reduc_traces, reduc_info = self._dimReducFn(traces)
    assert len(traces) == len(dim_reduc_traces), (f"{len(traces) = } !="
                                                  f"{len(dim_reduc_traces) = }")
    trial_traces_reduc = {
            trial_num: reduc_trace
            for trial_num, reduc_trace in zip(trial_traces, dim_reduc_traces)}
    assert len(trial_traces_reduc) == len(trial_traces)
    # Save the traces their trials
    sess_trial_traces = self._sess_trials_traces_res.get(sess_name, {})
    for trial_num, trial_trace in trial_traces_reduc.items():
      trial_traces = sess_trial_traces.get(trial_num, {})
      trial_traces[trace_id] = trial_trace
      sess_trial_traces[trial_num] = trial_traces
    self._sess_trials_traces_res[sess_name] = sess_trial_traces
    # Now save the reduc info
    ses_reduc_info = self._sess_trace_reduc_info.get(sess_name, {})
    ses_reduc_info[trace_id] = reduc_info
    self._sess_trace_reduc_info[sess_name] = ses_reduc_info
    return df

  def dimReducName(self):
    return self._dimReducFn('name')

  def descr(self):
    return f"Running {self.dimReducName()}"
