
def calcWinLoseUpdates(df, col_prefix=""):
    def _calcUpdate(df, col_prefix):
        if not len(df):
            return 0
        update = 100*(df[f"{col_prefix}Stay"].sum() -
                      df[f"{col_prefix}StayBaseline"].sum())/len(df)
        return update
    prev_win_df = df[df[f"{col_prefix}PrevOutcomeCount"] >= 1]
    prev_lose_df = df[df[f"{col_prefix}PrevOutcomeCount"] <= -1]
    win_stay_update = _calcUpdate(prev_win_df, col_prefix)
    lose_stay_update = _calcUpdate(prev_lose_df, col_prefix)
    return win_stay_update, len(prev_win_df), \
           lose_stay_update, len(prev_lose_df)