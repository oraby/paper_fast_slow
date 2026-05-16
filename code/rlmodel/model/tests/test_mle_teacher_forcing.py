import numpy as np
import pandas as pd

from ..mle import MLEModelConfig, evaluate_neg_loglik


def _df_two_trials():
    return pd.DataFrame([
        dict(Name="S1", Date=pd.Timestamp("2026-01-01"), SessionNum=1,
             TrialNumber=1, SessId="S1_2026-01-01_1", DV=1.0, DVstr="1",
             valid=True, calcStimulusTime=0.12, ChoiceLeft=0.0,
             ChoiceCorrect=1.0),
        dict(Name="S1", Date=pd.Timestamp("2026-01-01"), SessionNum=1,
             TrialNumber=2, SessId="S1_2026-01-01_1", DV=1.0, DVstr="1",
             valid=True, calcStimulusTime=0.12, ChoiceLeft=1.0,
             ChoiceCorrect=0.0),
    ])


def test_mle_q_update_uses_observed_choice_not_model_direction():
    params = {
        "DRIFT_COEF": 2.0,
        "NOISE_SIGMA": 1.0,
        "BOUND": 1.0,
        "NON_DECISION_TIME": 0.02,
        "ALPHA": 0.5,
        "BIAS_COEF": 0.5,
        "Q_VAL_OFFSET": 0.0,
    }
    config = MLEModelConfig(
        drift_fn_str="Classic",
        bias_fn_str="Q-Val",
        noise_fn_str="Normal(0, 1)",
        include_Q=True,
        include_RewardRate=False,
        dt=0.01,
        t_dur=0.2,
        dx=0.1,
    )

    result = evaluate_neg_loglik(params, _df_two_trials(), config, return_df=True)

    # Trial 1 has positive DV but observed right choice. Teacher forcing must
    # update Q_R, so trial 2 sees Q_R=0.75 and Q_L still 0.5.
    trial2 = result.mle_df[result.mle_df.TrialNumber == 2].iloc[0]
    assert np.isclose(trial2.mle_Q_left_before, 0.5)
    assert np.isclose(trial2.mle_Q_right_before, 0.75)
    assert trial2.mle_Q_rel_before < 0
    assert np.isfinite(result.neg_loglik)
