
# from .bias import _biasFixed, _biasMean, _biasQVal
# from .noise import _noiseNormal, _noiseQval

# def readParams(dict_params, drift_fn_str, include_Q, include_RewardRate):
#     # clear_output(wait=True)
#     # fig, axs = _createFig(fig)
#     biasFn = dict_params["Bias Fn"]
#     if biasFn == _biasFixed:
#         biasFn_kwargs = {"offset": dict_params["Bias Fixed"],
#                          "bias_affinity": dict_params["Bias Affinity"]}
#         plot_bias_dir = True
#     elif biasFn == _biasMean:
#         biasFn_kwargs = {"mean": dict_params["Bias_mu"],
#                          "sigma": dict_params["Bias_sigma"],
#                          "bias_affinity": dict_params["Bias Affinity"]}
#         plot_bias_dir = True
#     elif biasFn == _biasQVal:
#         biasFn_kwargs = {}
#         plot_bias_dir = False
#     else:
#         raise ValueError(f"Unknown bias function: {biasFn}")

#     copy_q_val = False
#     driftFn = dict_params["Drift Fn"]
#     driftFn_args = []
#     driftFn_df_cols = []
#     if include_RewardRate:
#         driftFn_df_cols += ["RewardRate"] #, "DriftRewardRateCoef"]
#     if drift_fn_str in ("Classic+Q", "Fixed Q-RewardRate", "Drift Q-RewardRate"):
#         driftFn_df_cols += ["Q_val"]
#         driftFn_args += ["Q_VAL_COEF"]
#         if drift_fn_str in ("Fixed Q-RewardRate", "Drift Q-RewardRate"):
#             driftFn_args += ["Q_VAL_DECAY_RATE"]
#         # if drift_fn_str == "Drift Q-RewardRate":
#         #     driftFn_df_cols += ["Q_L", "Q_R", #"BOUND"
#         #                         ]
#         copy_q_val = True

#     noiseFn = dict_params["Noise Fn"]
#     if noiseFn == _noiseNormal:
#         noiseFn_df_cols = []
#     elif noiseFn == _noiseQval:
#         noiseFn_df_cols = ["Q_val", "Q_VAL_DECAY_RATE", "Q_VAL_COEF"]
#         copy_q_val = True

#     if copy_q_val:
#         assert include_Q
#         plot_bias_dir = False # TODO: Fix this properly and pas the enum

#     fixed_params = dict(biasFn=biasFn,
#                         biasFn_kwargs=biasFn_kwargs,
#                         driftFn=driftFn,
#                         driftFn_df_cols=driftFn_df_cols,
#                         driftFn_args=driftFn_args,
#                         noiseFn=noiseFn,
#                         noiseFn_df_cols=noiseFn_df_cols,
#     )
#     return fixed_params, plot_bias_dir

