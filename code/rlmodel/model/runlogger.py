import pandas as pd

class RunLogger():
    def __init__(self):
        self._run_dict = {}

    def __getattr__(self, key):
        return self._run_dict[key]

    def __setattr__(self, key, value):
        if key.startswith("_"):
            self.__dict__[key] = value
            return
        run_li = self._run_dict.get(key, [])
        run_li.append(value)
        self._run_dict[key] = run_li

    def toDF(self):
        expanded_dict = {key: [] for key in self._run_dict.keys()}
        # Get first the number of items per each entry, we will use dx
        # as a reference
        num_entries_li = [len(v) for v in self._run_dict["dx"]]
        # Expand list of lists to a single list, and expand entries with a
        # single value to a list of the same length as the other entries
        for key, val_li in self._run_dict.items():
            is_li = hasattr(val_li[0], "__len__")
            expanded_val_li = []
            for idx, val in enumerate(val_li):
                try:
                    if is_li:
                        assert len(val) == num_entries_li[idx], (
                            f"{len(val) = } != {num_entries_li[idx] = } - Error in key: {key}, idx: {idx}")
                        expanded_val_li += list(val)
                    else:
                        expanded_val_li += [val] * num_entries_li[idx]
                except:
                    print(f"Error in key: {key}, idx: {idx}")
                    raise
            expanded_dict[key] = expanded_val_li
        return pd.DataFrame(expanded_dict)

    def clear(self):
        self._run_dict = {}


# run_logger = RunLogger()
