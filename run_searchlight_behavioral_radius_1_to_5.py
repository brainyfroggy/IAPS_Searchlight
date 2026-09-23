"""Run the analysis cells from searchlight_behavioral_radius_1_to_5.ipynb."""

import json
from pathlib import Path


NOTEBOOK_PATH = Path(
    "N:/Experimental_Data/yujunchen/projects/IAPS_Searchlight/"
    "searchlight_behavioral_radius_1_to_5.ipynb"
)
ANALYSIS_CELL_INDICES = [1, 2, 3, 5, 7]


notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
analysis_source = "\n\n".join(
    "".join(notebook["cells"][index]["source"])
    for index in ANALYSIS_CELL_INDICES
)
exec(compile(analysis_source, str(NOTEBOOK_PATH), "exec"), {})
