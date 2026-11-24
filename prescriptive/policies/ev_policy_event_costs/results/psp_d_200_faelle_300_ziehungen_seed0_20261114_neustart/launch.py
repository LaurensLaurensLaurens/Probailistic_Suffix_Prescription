"""Launch the prepared event-cost rerun independently of the tool session."""
import json
import os
from pathlib import Path
import subprocess
import sys

out = Path(__file__).resolve().parent
root = out.parents[4]
if (out / "pipeline.pid").exists():
    raise FileExistsError("Already launched; inspect status instead of starting twice")
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
with (out / "pipeline.log").open("x") as stream:
    process = subprocess.Popen([sys.executable, "-u", str(out / "run.py")], cwd=root,
                               env=env, stdin=subprocess.DEVNULL, stdout=stream,
                               stderr=subprocess.STDOUT, start_new_session=True)
(out / "pipeline.pid").write_text(str(process.pid) + "\n")
print(json.dumps({"pid": process.pid, "output": str(out)}, ensure_ascii=False))
