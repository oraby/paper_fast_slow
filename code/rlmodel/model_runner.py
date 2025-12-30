

# %%
# Run this to enable loading from relative packes
if "PKG" not in globals():
    import importlib, importlib.util, sys, pathlib # https://stackoverflow.com/a/50395128/11996983
    # PKG = %pwd
    PKG = pathlib.Path(".").resolve()
    root_parent_level = 2
    root = PKG
    full_pkg = f"{root.name}"
    for _ in range(root_parent_level):
      root = root.parent
      full_pkg = f"{root.name}.{full_pkg}"
      MODULE_PATH = f"{root}{pathlib.os.path.sep}__init__.py"
      MODULE_NAME = f"{root.name}"
      spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
      module = importlib.util.module_from_spec(spec)
      sys.modules[spec.name] = module
      spec.loader.exec_module(module)
    __package__ = full_pkg



from .model.initvals import NUM_CPUS, InitVals, DT, T_dur
from .model import fit
from .model.drift import DRIFT_FN_DICT
from .model.bias import BIAS_FN_DICT
from .model.noise import NOISE_FN_DICT
from ...behavior.util.assigndvstr import assignDVStr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import pickle

# %%

plt.rcParams['svg.fonttype'] = 'none'

DF_FP = "../../paper_fast_slow/data/behavior/df_behavior_full.pkl"

def mergeDF(df_full, df_filtered):
    df_filtered_sess = df_filtered[["Name", "Date", "SessionNum"]]
    df_sess__keys = df_filtered_sess.value_counts().index
    df_li = []
    for sess_key in df_sess__keys:
        # print("Sess keys: ", sess_key)
        name, date, sess_num = sess_key
        df_sess = df_full[(df_full["Name"] == name) &
                        (df_full["Date"] == date) &
                        (df_full["SessionNum"] == sess_num)]
        if df_sess.DV.isnull().any():
            print(f"Nan in DV: {df_sess.DV.isnull().sum()}/{len(df_sess)}")
            continue
        df_sess = df_sess.sort_values("TrialNumber")
        df_li.append(df_sess)
    df = pd.concat(df_li).reset_index(drop=True)
    return df

def loadMerged():
    _FP2 = "../../paper_fast_slow/data/behavior/df_behavior.pkl"
    df_behavior = pd.read_pickle(_FP2)
    df_behavior_full = pd.read_pickle(r"/home/main/OnedriveFloatingPersonal/BpodUser/Data/Aggregates/AllAnimals__2024_01_04.pkl")
    df_behavior = mergeDF(df_behavior_full, df_behavior)
    df_behavior.to_pickle(DF_FP)
    return df_behavior

# df_behavior = loadMerged()

def loadDF(min_valid_trials, accepts_subjects=[]):
    df_behavior = pd.read_pickle(DF_FP)

    df_ewd = df_behavior[df_behavior.EarlyWithdrawal == 1]
    # print(df_ewd.ChoiceCorrect.isnull().sum(), df_ewd.ChoiceLeft.notnull().sum())
    df_behavior.loc[df_ewd.index, "ChoiceLeft"] = np.random.choice([0, 1], size=len(df_ewd), p=[0.5, 0.5])
    df_behavior["ChoiceLeft"] = df_behavior["ChoiceLeft"].astype(float)
    df_behavior["ChoiceCorrect"] = df_behavior["ChoiceCorrect"].astype(float)

    df_behavior = assignDVStr(df_behavior)
    print(f"Nullifying: {(df_behavior.calcStimulusTime > T_dur).sum():,}/{len(df_behavior):,} trial")
    df_behavior.loc[df_behavior.calcStimulusTime > T_dur, "ChoiceCorrect"] = np.nan # Treat as no choice
    df_behavior.loc[df_behavior.calcStimulusTime > T_dur, "calcStimulusTime"] = np.nan # Treat as no decision time
    accepted_subjecteds = []
    for name, subject_df in df_behavior.groupby("Name"):
        subject_df_valid = subject_df[subject_df.valid]
        if len(subject_df_valid) < min_valid_trials and \
           name not in accepts_subjects:
            print(f"Removing: {name} with {len(subject_df):,} trials")
            continue
        accepted_subjecteds.append(name)
    df_behavior = df_behavior[df_behavior.Name.isin(accepted_subjecteds)]
    return df_behavior


def runModel(df, bias_fn_str, drift_fn_str, noise_fn_str, is_loss_no_dir,
             evolve_res : dict = None, num_cpus=NUM_CPUS, dry_run=False):
    biasFn = BIAS_FN_DICT[bias_fn_str]
    driftFn = DRIFT_FN_DICT[drift_fn_str]
    noiseFn = NOISE_FN_DICT[noise_fn_str]
    if evolve_res is None:
        evolve_res = {}
    
    init_vals = InitVals()
    init_vals_dict = init_vals.toDict()

    evolve_res_res = fit.simulateDDM(df,
                                     driftFn=driftFn, 
                                     biasFn=biasFn,
                                     noiseFn=noiseFn,
                                     bounds_and_defaults=init_vals_dict,
                                     dt=DT, t_dur=T_dur,
                                     is_loss_no_dir=is_loss_no_dir,
                                     num_cpus=num_cpus,
                                     evolvs_res=evolve_res,
                                     dry_run=dry_run)
    evolve_res.update(evolve_res_res)
    return evolve_res


def runTest():
    df_behavior = loadDF()
    noise_fn_str = "Normal(0, 1)"
    for drift_fn_str in DRIFT_FN_DICT:
        for bias_fn_str in BIAS_FN_DICT:
            print("Drift:", drift_fn_str, "Bias:", bias_fn_str)
            runModel(df_behavior, drift_fn_str=drift_fn_str, 
                     bias_fn_str=bias_fn_str, noise_fn_str=noise_fn_str,
                     num_cpus=1, dry_run=True)
            print()


def main():
    # Add command line arguments to read drift_fn_str, bias_fn_str, 
    # noise_fn_str, num_cpus, dry_run
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift", type=str, required=True,
                        choices=DRIFT_FN_DICT.keys())
    parser.add_argument("--bias", type=str, required=True,
                        choices=BIAS_FN_DICT.keys())
    parser.add_argument("--noise", type=str, #required=True,
                        choices=NOISE_FN_DICT.keys(), default="Normal(0, 1)")
    parser.add_argument("--loss-no-dir", action="store_true", default=False,
                        help="Calculate Loss without direction")
    parser.add_argument("--num-cpus", type=int, default=NUM_CPUS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--load-evolve", action="store_true")
    parser.add_argument("--remove-subject", type=str, default=None,
                        action='append' )
    args = parser.parse_args()
    if args.test:
        runTest()
        return
    
    df_behavior = loadDF(min_valid_trials=0)
    df_behavior = _extendTrials(df_behavior)
    # df_behavior = loadDF(min_valid_trials=3000, accepts_subjects=["Avgat1"])
    evolve_res = None
    if args.load_evolve:
        load_evolve_fp = fit.evolveFP(args.drift, args.bias, args.noise,
                                      t_dur=T_dur, dt=DT, 
                                      is_loss_no_dir=args.loss_no_dir)
        assert load_evolve_fp.exists(), f"File not found: {load_evolve_fp}"
        with open(load_evolve_fp, "rb") as f:
            evolve_res = pickle.load(f)
    if args.remove_subject:
        for subject in args.remove_subject:
            assert subject in evolve_res, f"Subject not found: {subject}"
        print("Removing subjects iteratively")
        # Run in a loop such that we remvoe one subject at a time
        for subject in args.remove_subject:
            print("Removing subject:", subject)
            evolve_res.pop(subject)
            evolve_res = runModel(df_behavior, bias_fn_str=args.bias, drift_fn_str=args.drift, 
                                  noise_fn_str=args.noise, num_cpus=args.num_cpus, 
                                  dry_run=args.dry_run, evolve_res=evolve_res,
                                  is_loss_no_dir=args.loss_no_dir)
    else:
        runModel(df_behavior, bias_fn_str=args.bias, drift_fn_str=args.drift, 
                 noise_fn_str=args.noise, num_cpus=args.num_cpus, 
                 dry_run=args.dry_run, evolve_res=evolve_res,
                 is_loss_no_dir=args.loss_no_dir)



def _extendTrials(df):
    # df = df.copy()
    sess_li = []
    subjects = df.Name.unique()
    for subject in subjects:
        df_subj = df[df.Name == subject]
        max_trial = df_subj.TrialNumber.max()
        for sess_key, sess_df in df_subj.groupby(["Date", "SessionNum"]):
            sess_li.append(sess_df)
            sess_max_trial = sess_df.TrialNumber.max()
            remaining_trials = max_trial - sess_max_trial
            if remaining_trials == 0:
                continue
            # Duplicate the last trial to fill the remaining trials
            # print("Subject:", subject, "Session:", sess_key, sess_max_trial, "Remaining trials:", remaining_trials)
            # assert len(sess_df) >= remaining_trials, "Didn't implement the case where there are more remaining trials than the session has"
            remaining_cpy = sess_df.iloc[-1].copy()
            remaining_cpy["valid"] = False
            li = []
            for i in range(remaining_trials):
                remaining_cpy["TrialNumber"] = sess_max_trial + i + 1
                li.append(remaining_cpy.copy())
            remaining_df = pd.DataFrame(li)
            sess_li.append(remaining_df)
            
    df = pd.concat(sess_li).reset_index(drop=True)
    df = df.sort_values(["Name", "Date", "SessionNum", "TrialNumber"])
    DEBUG = False
    if DEBUG:
        for subject in subjects:
            df_subj = df[df.Name == subject]
            max_trial = df_subj.TrialNumber.max()
            print("Subject:", subject, "Max Trial:", max_trial)
            for sess_key, sess_df in df_subj.groupby(["Date", "SessionNum"]):
                print("   Subject:", subject, "Session:", sess_key, "Trials:", len(sess_df), 
                    "Sess Max Trial:", sess_df.TrialNumber.max())
                sess_df = sess_df[sess_df.valid]
                print("   Max valid Trials:", sess_df.TrialNumber.max())
    
    ##
    # [print("col: ", col, "dtype: ", df_behavior[col].dtype) for col in df_behavior.columns]
    drop_cols = []
    df["Name"] = df["Name"].astype("string")
    df["DVstr"] = df["DVstr"].astype("string")
    df["Date"] =  pd.to_datetime(df["Date"])
    for col in df.columns:
        if df[col].dtype == "object":
            drop_cols.append(col)
        elif df[col].dtype == np.float64:
            # print("col: ", col,)
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype != np.float32:
            print("col: ", col, "dtype: ", df[col].dtype)
    print("Dropping: ", drop_cols)
    df = df.drop(columns=drop_cols)

    return df

if __name__ == "__main__":
    main()