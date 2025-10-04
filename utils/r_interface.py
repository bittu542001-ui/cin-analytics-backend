# utils/r_interface.py
import subprocess
import json
import tempfile
import os
from typing import Dict, Any

def run_r_script(script_path: str, input_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run an R script with one argument: path to a JSON input file.
    The R script should print JSON to stdout (or write to a file and print path).
    Returns a dict with keys: error(bool), rc(int), stdout, stderr, result(dict if JSON).
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    try:
        tmp.write(json.dumps(input_obj, default=str).encode("utf-8"))
        tmp.flush()
        tmp.close()
        proc = subprocess.run(["Rscript", script_path, tmp.name], capture_output=True, text=True)
        out = {"error": False, "rc": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
        if proc.returncode != 0:
            out["error"] = True
            return out
        # try parse stdout as json
        try:
            out_json = json.loads(proc.stdout)
            out["result"] = out_json
        except Exception:
            out["result"] = {"raw_stdout": proc.stdout.strip()}
        return out
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
