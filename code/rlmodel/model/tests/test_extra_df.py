"""Tests for ``--extra-df`` / ``loadDF(extra_dfs=...)``.

Supplemental behavior dataframes are concatenated onto the main one at the
TOP of ``loadDF`` (``_concatExtraDFs``), so they receive the identical
cleaning pipeline. Each extra must carry every column the main dataframe
has, so a missing one fails at startup instead of silently concatenating
as all-NaN.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ...model_runner import _concatExtraDFs, loadDF


def _df(names, extra_cols=None):
    df = pd.DataFrame({
        "Name": names,
        "TrialNumber": range(1, len(names) + 1),
        "valid": True,
    })
    for col, val in (extra_cols or {}).items():
        df[col] = val
    return df


# ---------------------------------------------------------------------------
# _concatExtraDFs
# ---------------------------------------------------------------------------

def test_no_extras_is_identity():
    df = _df(["S1", "S2"])
    # Falsy in every form the CLI / callers can produce → same object back.
    assert _concatExtraDFs(df, None) is df
    assert _concatExtraDFs(df, []) is df


def test_concat_dataframe_appends_rows_and_subjects():
    main = _df(["S1", "S1", "S2"])
    extra = _df(["H1", "H2"])
    out = _concatExtraDFs(main, extra)          # bare DataFrame, not a list
    assert len(out) == len(main) + len(extra)
    assert set(out["Name"]) == {"S1", "S2", "H1", "H2"}
    # Fresh 0..n-1 index so downstream .loc[mask, col] writes are unambiguous.
    assert list(out.index) == list(range(len(out)))
    # The input is not mutated.
    assert len(main) == 3


def test_concat_multiple_extras_in_order():
    main = _df(["S1"])
    out = _concatExtraDFs(main, [_df(["H1"]), _df(["H2"])])
    assert list(out["Name"]) == ["S1", "H1", "H2"]


def test_concat_from_pickle_path(tmp_path):
    main = _df(["S1"])
    fp = tmp_path / "extra.pkl"
    _df(["H1", "H2"]).to_pickle(fp)
    # Path object and str spelling both work, as does the single-item form.
    assert list(_concatExtraDFs(main, fp)["Name"]) == ["S1", "H1", "H2"]
    assert list(_concatExtraDFs(main, [str(fp)])["Name"]) == ["S1", "H1", "H2"]


def test_missing_column_raises_naming_it():
    main = _df(["S1"], extra_cols={"ChoiceLeft": 1.0, "DVstr": "a"})
    extra = _df(["H1"], extra_cols={"ChoiceLeft": 1.0})   # no DVstr
    with pytest.raises(ValueError, match="DVstr"):
        _concatExtraDFs(main, extra)


def test_extra_columns_are_kept_as_nan_for_main_rows():
    main = _df(["S1"])
    extra = _df(["H1"], extra_cols={"Age": 42})
    out = _concatExtraDFs(main, extra)
    assert "Age" in out.columns
    assert out.loc[out.Name == "S1", "Age"].isnull().all()
    assert (out.loc[out.Name == "H1", "Age"] == 42).all()


def test_missing_file_raises_value_error(tmp_path):
    # ValueError (not FileNotFoundError) so main() can route it to
    # parser.error alongside the other startup validations.
    with pytest.raises(ValueError, match="not found"):
        _concatExtraDFs(_df(["S1"]), tmp_path / "nope.pkl")


def test_non_dataframe_pickle_raises(tmp_path):
    fp = tmp_path / "bad.pkl"
    pd.Series([1, 2, 3]).to_pickle(fp)
    with pytest.raises(ValueError, match="expected a DataFrame"):
        _concatExtraDFs(_df(["S1"]), fp)


def test_overlapping_subject_names_warn_but_merge(capsys):
    main = _df(["S1"])
    out = _concatExtraDFs(main, _df(["S1", "H1"]))
    assert "WARNING" in capsys.readouterr().out
    # Merged, not dropped: the duplicate name becomes one longer subject.
    assert len(out[out.Name == "S1"]) == 2


# ---------------------------------------------------------------------------
# loadDF integration — the extras must go through the SAME cleaning
# ---------------------------------------------------------------------------

def _behavior_df(names, stim_times):
    """Minimal frame with the columns loadDF's cleaning touches."""
    return pd.DataFrame({
        "Name": names,
        "Date": "2020-01-01",
        "SessionNum": 1,
        "TrialNumber": range(1, len(names) + 1),
        "calcStimulusTime": stim_times,
        "ChoiceLeft": 1.0,
        "ChoiceCorrect": 1.0,
        "valid": True,
    })


def test_loadDF_cleans_the_extra_rows_too(tmp_path):
    """A too-long trial in the SUPPLEMENTAL df must be invalidated exactly
    like one in the main df — that is the point of concatenating before
    the cleaning rather than after it."""
    main_fp, extra_fp = tmp_path / "main.pkl", tmp_path / "extra.pkl"
    _behavior_df(["S1", "S1"], [0.5, 99.0]).to_pickle(main_fp)
    _behavior_df(["H1", "H1"], [0.5, 99.0]).to_pickle(extra_fp)

    df = loadDF(df_fp=main_fp, extra_dfs=[extra_fp])
    assert set(df["Name"]) == {"S1", "H1"}
    # The 99s-stimulus trial is invalid for BOTH the main and extra subject.
    assert list(df.groupby("Name")["valid"].sum()) == [1, 1]
    # And the result is sorted by the usual keys.
    assert list(df["Name"]) == sorted(df["Name"])


def test_loadDF_min_valid_trials_applies_to_extra_subjects(tmp_path):
    main_fp, extra_fp = tmp_path / "main.pkl", tmp_path / "extra.pkl"
    _behavior_df(["S1", "S1", "S1"], [0.5] * 3).to_pickle(main_fp)
    _behavior_df(["H1"], [0.5]).to_pickle(extra_fp)
    # H1 has 1 valid trial < 2 → filtered out, same rule as the main df's.
    df = loadDF(min_valid_trials=2, df_fp=main_fp, extra_dfs=[extra_fp])
    assert set(df["Name"]) == {"S1"}
    # ...unless explicitly accepted.
    df = loadDF(min_valid_trials=2, accepts_subjects=["H1"],
                df_fp=main_fp, extra_dfs=[extra_fp])
    assert set(df["Name"]) == {"S1", "H1"}
