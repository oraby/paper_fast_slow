
def assignDVStr(df, col="DV", res_col="DVstr"):
  def DVstr(row):
    dv = abs(row[col])
    dv = "Easy" if dv >= 0.66 else ("Hard" if dv <= 0.33 else "Med")
    return dv
  df[res_col] = df.apply(DVstr, axis=1)
  return df
