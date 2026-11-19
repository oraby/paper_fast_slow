import numpy as np
import pandas as pd
from collections import namedtuple

SplitResult = namedtuple("SplitResult", ["df", "val", "label"])

def conditionsCorrectIncorrect(df):
    return ("ChoiceCorrect",
            SplitResult(df=df[df.ChoiceCorrect == True], val=True,
                        label="Correct"),
            SplitResult(df=df[df.ChoiceCorrect == False], val=False,
                        label="Incorrect"))

def conditionsPrevCorrectIncorrect(df):
    return ("PrevChoiceCorrect",
            SplitResult(df=df[df.PrevChoiceCorrect == True], val=True,
                        label="PrevCorrect"),
            SplitResult(df=df[df.PrevChoiceCorrect == False], val=False,
                        label="PrevIncorrect"))

def conditionsEasyDifficult(df):
    dv_abs_prcnt = df.DV.abs()*100
    easy_df = df[np.isclose(dv_abs_prcnt, df.Difficulty1)].copy()
    easy_df["IsEasy"] = True
    hard_df = df[np.isclose(dv_abs_prcnt, df.Difficulty3) |
                             np.isclose(dv_abs_prcnt, df.Difficulty4)].copy()
    hard_df["IsEasy"] = False
    return ("IsEasy",
            SplitResult(df=easy_df, val=True, label="Easy"),
            SplitResult(df=hard_df, val=False, label="Difficult"))

def conditionsPrevEasyDifficult(df):
    prev_dv_abs_prcnt = df.PrevDV.abs()*100
    prev_easy_df = df[np.isclose(prev_dv_abs_prcnt, df.Difficulty1)].copy()
    prev_easy_df["PrevIsEasy"] = True
    prev_hard_df = df[np.isclose(prev_dv_abs_prcnt, df.Difficulty3) |
                      np.isclose(prev_dv_abs_prcnt, df.Difficulty4)].copy()
    prev_hard_df["PrevIsEasy"] = False
    return ("PrevIsEasy",
            SplitResult(df=prev_easy_df, val=True, label="PrevEasy"),
            SplitResult(df=prev_hard_df, val=False, label="PrevDifficult"))

def conditionsLeftRight(df):
    return ("ChoiceLeft",
            SplitResult(df=df[df.ChoiceLeft == True], val=True, label="Left"),
            SplitResult(df=df[df.ChoiceLeft == False], val=False,label="Right"))

def conditionsPrevLeftRight(df):
    return ("PrevChoiceLeft",
            SplitResult(df=df[df.PrevChoiceLeft == True], val=True,
                        label="PrevLeft"),
            SplitResult(df=df[df.PrevChoiceLeft == False], val=False,
                        label="PrevRight"))
