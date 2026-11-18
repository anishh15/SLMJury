"""The order in which the three judging-protocol runners visit the datasets.

A result file is written only after a whole dataset is judged, so a stopped run keeps only finished datasets.
math goes first, the two largest (hellaswag, gsm_plus, about 10,000 items each) last; unlisted ones sort after them.
"""

DATASET_ORDER = [
    "math",            #  5,000
    "truthfulqa",      #    684
    "arc_challenge",   #  1,172
    "winogrande",      #  1,267
    "gsm8k",           #  1,319
    "arc_easy",        #  2,376
    "hellaswag",       # 10,042
    "gsm_plus",        # 10,552
]

_RANK = {name: i for i, name in enumerate(DATASET_ORDER)}


def dataset_sort_key(stem: str):
    """Sort key for a dataset name; unknown names sort after the known ones."""
    return (_RANK.get(stem, len(DATASET_ORDER)), stem)


def ordered_solution_files(student_dir):
    """The *.json solution files in ``student_dir``, in DATASET_ORDER."""
    return sorted(student_dir.glob("*.json"), key=lambda p: dataset_sort_key(p.stem))
