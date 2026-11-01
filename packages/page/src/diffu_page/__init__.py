"""Archive pages as data and the page pipeline, in layers that each import only the ones before them: :mod:`page` (a
page as data), :mod:`captions` (a crop's prompt), :mod:`quality` (a drawing read back and judged), :mod:`draw` (planning
and drawing), :mod:`data` (the training and evaluation sets) and :mod:`cli`. Every layer uses :mod:`output`,
:mod:`settings` and :mod:`local_run`."""
