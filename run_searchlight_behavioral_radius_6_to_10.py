"""Run all or one selected radius from the radius 6–10 notebook."""

import json
import sys
from pathlib import Path


NOTEBOOK_PATH = Path(
    "N:/Experimental_Data/yujunchen/projects/IAPS_Searchlight/"
    "searchlight_behavioral_radius_6_to_10.ipynb"
)

notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
namespace = {}
for cell_index in [1, 2, 3]:
    source = "".join(notebook["cells"][cell_index]["source"])
    exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)

if len(sys.argv) > 1:
    radius = int(sys.argv[1])
    if radius not in range(6, 11):
        raise ValueError("Radius must be between 6 and 10")
    namespace["RADII_VOXELS"] = [radius]
    # Multiple radii may run concurrently; keep model evaluation in-process
    # to avoid oversubscribing joblib workers.
    namespace["N_JOBS"] = 1

for cell_index in [5, 7]:
    source = "".join(notebook["cells"][cell_index]["source"])
    exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)
