from scipy import stats

def normalizeSTAcrossSubjects(df):
    def normalizeST(df):
        df = df.copy()
        st_data_cols = ["calcStimulusTime"] + [col for col in df.columns
                                      if col.startswith("PrevCalcStimulusTime")]
        st_data_vals = [df[col].values for col in st_data_cols]
        # PrevCalcStimulusTime1, PrevCalcStimulusTime2, ...
        # trnasformed_data = stats.mstats.winsorize(trnasformed_data,
        #                                           limits=[0.05, 0.05])
        # from sklearn.preprocessing import RobustScaler as transformer
        # from sklearn.preprocessing import QuantileTransformer as transformer
        # transformed_data = transformer().fit_transform(
        #                                       trnasformed_data.reshape(-1, 1))
        # transformed_data, lmbd = stats.boxcox(trnasformed_data)
        transformed_data_vals = [stats.zscore(col_data, nan_policy="omit")
                                 for col_data in st_data_vals]
        # transformed_data = np.log(trnasformed_data)
        #
        # Save calcStimulusTime as transformedCalcStimulusTime
        for col, transformed_data in zip(st_data_cols, transformed_data_vals):
            col_upper = col[0].upper() + col[1:]
            df[f"transformed{col_upper}"] = transformed_data
        return df

    df = df.groupby(["Name"],#, "Date", "SessionNum"],
                    group_keys=False).apply(normalizeST)
    return df
