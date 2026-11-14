"""Curating real archive scans with machine-made labels (the pools) into training sets.

A pool is a folder of volumes of ALTO files an old pipeline wrote, beside a folder of their scans. Each page of a pool
is a row of one Lance table (:mod:`diffu_curate.table`) that the checks add their scores to: what its ALTO file alone
says (:mod:`diffu_curate.page_features`), and, measured on the scans the ground truth also holds, how wrong such labels
are (:mod:`diffu_curate.calibration`). Checks score pages; nothing is dropped until a training set is selected, so a
threshold is set, and reset, without scoring again.
"""
