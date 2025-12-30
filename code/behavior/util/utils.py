def bootStrap(df, *, grps_sizes_li, with_replacement):
  """Create permutations of the data in the dataframe
  """
  df_li = []
  for size in grps_sizes_li:
    df_li.append(df.sample(size, replace=with_replacement))
  return df_li

def chunks(lst, n):
  """Yield successive n-sized chunks from lst.
  Copied from: https://stackoverflow.com/a/312464
  """
  for i in range(0, len(lst), n):
      yield lst[i:i + n]

def sideBySideCmp(df):
  '''This examples detects repeated TrialNumber for same sessions (i.e bug)'''
  dup_df = df[df.duplicated(subset=("Name","Date", "SessionNum", "TrialNumber"),
                            keep=False)]
  #print("Duplicated len:", dup_df.Name.unique(), dup_df.Date.unique())
  for (name, date, session_num, trial_num), dup_entry in \
                  dup_df.groupby(["Name", "Date", "SessionNum", "TrialNumber"]):
    diff = dup_entry.iloc[0] == dup_entry.iloc[1]
    mismatch_col = diff[~diff].index.to_numpy()
    side_by_side = dup_entry[mismatch_col].transpose()
    side_by_side.dropna(how="all", inplace=True)
    print("Name:", name, "Date:", date, "Trial num:", trial_num,
          "\n", dup_entry.File.iloc[0], "\n", dup_entry.File.iloc[1])
          #side_by_side)
