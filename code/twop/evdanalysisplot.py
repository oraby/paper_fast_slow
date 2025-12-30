
from ..common.clr import Difficulty as DifficultyClr, Direction as DirectionClr

def AllClrFn(_id):
    return 'k'

def DirectionClrFn(_id):
    _ = '_' if "_Decision" in _id else ""
    if    f"{_}DecisionLeft"  in _id: return DirectionClr.Left
    elif  f"{_}DecisionRight" in _id: return DirectionClr.Right
    else:                             return "gray"

def ChoiceOutcomeClrFn(_id):
    if     "Correct" in _id: return 'g'
    elif "Incorrect" in _id: return 'r'
    else:                    return "gray"

def _stripId(_id):
    loc = _id.find("Sampling")
    if loc != -1:
        _id = _id[loc+len("Sampling"):]
    loc = _id.find("Port")
    if loc != -1:
        _id = _id[loc+len("Port"):]
    _id = _id.strip().strip('_').strip()
    return _id

Difficulty_clr = lambda _id: getattr(DifficultyClr, _stripId(_id))

Difficulty_Direction = lambda _id: getattr(DifficultyClr, _stripId(_id).rsplit("_",2)[0])

def legendLabelAndLineStyle(trace_id):
    ls = "solid"
    is_difficulty = any((diff in trace_id for diff in ["Easy", "Med", "Hard"]))
    if "Prev" in trace_id  and "Left" in trace_id and "Right" in trace_id:
        ls = "dotted"
    elif ("loorrect" in trace_id or is_difficulty)  and ("Left" in trace_id):
        ls = "dashed"
    trace_id = _stripId(trace_id)
    return trace_id, ls