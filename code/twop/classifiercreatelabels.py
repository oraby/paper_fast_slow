

def threeDifficultiesLabels(df, ignore_direction, separate_hard=True):
  df = df.copy()
  df["DVAsLabel"] = df.DVstr.astype("category").cat.codes + 1
  # Assign hard 0 so R/L hard dificulties are treated the same
  df.loc[df.DVstr == "Hard",  "DVAsLabel"] = 0
  df.loc[df.DVstr == "Medium",  "DVAsLabel"] = 1
  df.loc[df.DVstr == "Easy",  "DVAsLabel"] = 2
  if not ignore_direction:
    if separate_hard:
      df.DVAsLabel += 1
    left_rewarded = df.LeftRewarded.copy()
    left_rewarded[left_rewarded == 0] = -1
    df["DVAsLabel"] = df.DVAsLabel * left_rewarded
  # print(df.DVAsLabel.unique())
  return "DVAsLabel", df

def difficultyAsLabel(df, difficulty, left_or_right_or_none):
  assert difficulty in df.DVstr.unique()
  assert len(df.DVstr.unique()) > 1, "Only one difficulty found"
  pos_label_index = df.DVstr == difficulty
  if left_or_right_or_none == "Left":
    pos_label_index = pos_label_index & (df.LeftRewarded == 1)
    direction = "Left"
  elif left_or_right_or_none == "Right":
    pos_label_index = pos_label_index & (df.LeftRewarded == 0)
    direction = "Right"
  else:
    assert left_or_right_or_none is None,("Unknown side: "
                                          f"{left_or_right_or_none}")
    direction = ""
  col_label = f"is{difficulty}{direction}"
  df = df.copy()
  df[col_label] = pos_label_index
  # return , df
  return "ChoiceLeft", df

def dvAsLabel(df, ignore_direction, treat_abs_dv_above_val_as_zero=0):
  if ignore_direction or treat_abs_dv_above_val_as_zero > 0:
    col_label = "DVAsLabel"
    df = df.copy()
    if ignore_direction:
      df[col_label] = df.DV.abs()
    else:
      df[col_label] = df.DV
    if treat_abs_dv_above_val_as_zero > 0:
      trgt_rows = df.DV.abs() <= treat_abs_dv_above_val_as_zero
      df.loc[trgt_rows, col_label] = 0
  else:
    col_label = "DV"
  return col_label, df

def mouseChoiceLabel():
  return "DVAsLabel"

def rewardedSideLabel():
  return "LeftRewarded"

