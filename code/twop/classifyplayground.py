from ..common.definitions import BrainRegion
from ..behavior.util import splitdata
from ..pipeline import pipeline
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
# import multiprocess as multiprocessing

def modelMetrics(y_test, y_pred):
  # Model Accuracy: true positive & negative from overaall positive & negative
  # if all(y_test - y_test.astype(int) == 0): # We were told its a bad idea
  #   y_pred = np.round(y_pred) # Round to nearest integer
  accuracy = metrics.accuracy_score(y_test, y_pred)
  # Model Recall: true positive from actual positive (TP + FN)
  recall = metrics.recall_score(y_test, y_pred, average="weighted",
                                zero_division=0)
  # Model Precision: true positive from predicted positive (TP + FP)
  # print("Y test:", y_test)
  # print("Y pred:", y_pred)
  precision = metrics.precision_score(y_test, y_pred, average="weighted", 
                                      zero_division=0)
  # precision = 0
  return accuracy, recall, precision

def test(df, col_label):
  accuracies = []
  recalls = []
  precisions = []
  predictions = []
  labels = []
  for sess_info, sess_df in splitdata.grpBySess(df):
    (accuracy_dict, recall_dict, precision_dict, predictions_dict,
     labels_dict) = _processSession(sess_df, col_label=col_label)
    accuracies.append(accuracy_dict)
    recalls.append(recall_dict)
    precisions.append(precision_dict)
    predictions.append(predictions_dict)
    labels.append(labels_dict)
  return accuracies, recalls, precisions, predictions, labels

def _processSession(df, col_label):
  # df = df.copy()
  # df = df[df.ChoiceLeft.notnull()]
  # df["DVAslabel"] = df.DVstr.astype("category").cat.codes + 1
  # # Assign hard 0 so R/L are treated the same
  # df.loc[df.DVstr == "Hard",  "DVAslabel"] = 0
  # df.loc[df.DVstr == "Medium",  "DVAslabel"] = 1
  # df.loc[df.DVstr == "Easy",  "DVAslabel"] = 2
  # left_rewarded = df.LeftRewarded.copy()
  # left_rewarded[left_rewarded == 0] = -1
  # df["DVAslabel"] = df.DVAslabel * left_rewarded
  # print(df.DVAslabel.unique())
  max_val = df[col_label].value_counts().max()
  labels_weights = {label: max_val/np.sum(df[col_label]==label)
                    for label in df[col_label].unique()}
  labels_counts = {label: np.sum(df[col_label] == label)
                    for label in df[col_label].unique()}
  print("labels_counts:", labels_counts)
  print("labels_weights:", labels_weights)
  
  pool_res = []
  sub_cols = ["traces_sets", "sole_owner", "BrainRegion", "Layer", "ShortName", 
              "Name"]
  if "epochs_names" in df.columns:
    sub_cols +=   "epochs_names", "epochs_ranges"
  sub_cols.append(col_label) # Maintain columns order
  df = df[sub_cols]
  # num_workers = 16
  # num_workers = processes=multiprocessing.cpu_count()-1
  # with multiprocessing.Pool(num_workers) as pool:
  # call = pool.apply_async
  # print("Pool:", pool)
  def call(f, kwds):
    return f(**kwds)
  if True:
    # step = 0.1
    step = 1
    for i in np.arange(step, 1+step, step=step):
      promise = call(_wrapCall, kwds=dict(df=df, col_label=col_label, labels_weights=labels_weights,
                                          ratio_sampling=i, ratio_move_to_port=0, dict_key="sampling",
                                          state_offset=i))
      pool_res.append(promise)
    # step = 0.25
    # for i in np.arange(step, 1+step, step=step):
    #   promise = call(_wrapCall, kwds=dict(df=df, col_label=col_label, labels_weights=labels_weights,
    #                                       ratio_sampling=1, ratio_move_to_port=i, dict_key="to_port",
    #                                       state_offset=i))
    #   pool_res.append(promise)
    
    accuracies = {"sampling":{}, "to_port":{}}
    recalls = {"sampling":{}, "to_port":{}}
    precisions = {"sampling":{}, "to_port":{}}
    predictions = {"sampling":{}, "to_port":{}}
    test_labels = {"sampling":{}, "to_port":{}}
    for promise in pool_res:
      # res = promise.get()
      res = promise
      (accuracy, recall, precision, prediction_li, label_li, dict_key,
       state_offset)  = res
      accuracies[dict_key][state_offset] = accuracy
      recalls[dict_key][state_offset] = recall
      precisions[dict_key][state_offset] = precision
      predictions[dict_key][state_offset] = prediction_li
      test_labels[dict_key][state_offset] = label_li

  return accuracies, recalls, precisions, predictions, test_labels

def _wrapCall(df, col_label, labels_weights, ratio_sampling, ratio_move_to_port, dict_key, state_offset):
  ret = _runOnData(df, ratio_sampling=ratio_sampling, 
                   ratio_move_to_port=ratio_move_to_port,
                   col_label=col_label, labels_weights=labels_weights)
  return tuple(list(ret) + [dict_key, state_offset])

def _runOnData(df, ratio_sampling, ratio_move_to_port, col_label,
               labels_weights):
  if ratio_sampling < 1:
    assert ratio_move_to_port == 0
  if ratio_move_to_port:
    assert ratio_sampling == 1
  features = []
  labels = []
  # display(df)
  for trial_idx, row in df.iterrows():
    traces_set = pipeline.getRowTracesSets(row)["neuronal"]
    # assert row.epochs_names == ["Sampling", "Movement to Lateral Port"], (
    #                                                 f"Found {row.epochs_names}")
    # assert len(row.epochs_ranges) == 2, "Expected 2 epochs to be there"
    sampling_rng = row.epochs_ranges[0]
    move_port_rng = row.epochs_ranges[1]
    trace_start = sampling_rng[0]
    # if ratio_move_to_port:
    #   end_idx = int(np.round((move_port_rng[1] - move_port_rng[0]) * 
    #                 ratio_move_to_port))
    #   trace_end = sampling_rng[1] + end_idx + 1
    # else:
    #   assert ratio_sampling
    #   trace_end = int(np.round((trace_start + 
    #                (sampling_rng[1] - sampling_rng[0]) * ratio_sampling))) + 1
    trace_end = move_port_rng[-1] + 1
          
    # traces_avg = [trace_data.mean() for trace_data in traces_set.values()]
    # print("sampling_rng:", sampling_rng, "move_port:", move_port_rng)
    # print("Trace start:", trace_start, "- Trace end:", trace_end)
    cut_traces =  [trace_data[trace_start:trace_end] 
                   for trace_data in traces_set.values()]
    whole_traces = np.concatenate(cut_traces)
    features.append(whole_traces)
    # label = row.ChoiceLeft
    label = row[col_label]
    labels.append(label) 
  features = np.array(features)
  labels = np.array(labels)
  ex_row = df.iloc[0]
  # if features.shape[1] < 20:
  #   print("Skipping ", row.ShortName)
  #   print("Features shape:", features.shape)
  #   print("labels shape:", labels.shape)
  #   return
  br = str(BrainRegion(ex_row["BrainRegion"]))
  print(f"Brain Region: {br} - Layer: {ex_row['Layer']} - "
        f"Animal: {ex_row.Name} - Short-Name: {ex_row['ShortName']} - "
        f"Ratio Sampling: {ratio_sampling} + Ratio port: {ratio_move_to_port}")
  accuracy, recall, precision, predictions, labels = _runClassifier(
                features=features, labels=labels, labels_weights=labels_weights,
                plot=(ratio_sampling == 1 or ratio_move_to_port == 1))
  # chance_level =  100/len(np.unique(labels))
  # print(f"\tAccuracy: {accuracy*100:.4g}%\t\t(Chance Perf: {chance_level:.4g}%)"
  #       f"- Num Features: {features.shape[1]} - "
  #       f"Num Trials: {features.shape[0]}")
  return accuracy, recall, precision, predictions, labels

def _runClassifier(features, labels, labels_weights, num_runs=20, plot=False,
                   return_all=False):
  accs, recs, precs = [], [], []
  all_predections = []
  all_test_labels = []

  # for i in range(num_runs):
  i = 0
  while i < num_runs:
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels,  test_size=0.3, # 70% training and 30% test
        stratify=labels,
        #random_state=1
        ) 
    # labels_test = labels_test.shuffle()
    labels_pred = _LogisticRegressionPredict(features_train=features_train, 
                              labels_train=labels_train, 
                              features_to_predict=features_test,
                              labels_weights=labels_weights)
    # from .classifierpytorch import runNN
    # labels_pred = runNN(features_train=features_train, 
    #                     labels_train=labels_train, 
    #                     features_to_predict=features_test,
    #                     labels_weights=labels_weights,
    #                     plot=i==0 and plot)
    if labels_pred is None:
      continue
    i += 1
    # if i == num_runs - 1:
    #   print("labels_pred:  ", labels_pred)
    #   print("actual labels:", labels_test)
    acc, rec, prec = modelMetrics(y_test=labels_test, y_pred=labels_pred)
    accs.append(acc), recs.append(rec), precs.append(prec)
    all_predections.append(labels_pred)
    all_test_labels.append(labels_test)
  if not return_all:
    return (np.mean(accs), np.mean(recs), np.mean(precs),
            np.concatenate(all_predections), np.concatenate(all_test_labels))
  else:
    return accs, recs, precs, all_predections, all_test_labels

def _LogisticRegressionPredict(features_train, labels_train,
                               features_to_predict, labels_weights):
  # Run GLM instead
  from sklearn.linear_model import LogisticRegression
  clf = LogisticRegression(max_iter=1000)
  # from sklearn.neural_network import MLPClassifier
  # clf = MLPClassifier(solver='adam', alpha=1e-5, max_iter=100000,
  #                     hidden_layer_sizes=(30, 20, 20, 5))
  #Train the model using the training sets
  try:
    clf = clf.fit(features_train, labels_train)
  except ValueError:
    return None
  # print("reg.coef_:", clf.coef_)
  #Predict the response for test dataset
  predicted_labels = clf.predict(features_to_predict)
  return predicted_labels

def _SVMPredict(features_train, labels_train, features_to_predict,
                labels_weights):
  # Create a svm Classifier
  # clf = svm.SVC(kernel='rbf', class_weight=labels_weights, break_ties=True)
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler
  clf = make_pipeline(StandardScaler(), svm.SVC(kernel='rbf',
                                                class_weight=labels_weights,
                                                break_ties=True))
  #Train the model using the training sets
  try:
    clf = clf.fit(features_train, labels_train)
  except ValueError:
    return None
  #Predict the response for test dataset
  predicted_labels = clf.predict(features_to_predict)
  return predicted_labels