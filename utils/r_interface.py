# utils/r_interface.py
import subprocess
import json
import tempfile
import os

def run_r_script(script_path, input_json):
    """
    script_path: path to R script (on server)
    input_json: python dict will be written to a temp json file and passed to Rscript
    R script must accept one argument: path to input json and write JSON to stdout or to file.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    try:
        tmp.write(json.dumps(input_json).encode("utf-8"))
        tmp.flush()
        tmp.close()
        # call: Rscript script.R /tmp/input123.json
        proc = subprocess.run(["Rscript", script_path, tmp.name], capture_output=True, text=True, check=False)
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if proc.returncode != 0:
            return {"error": True, "stderr": stderr, "stdout": stdout, "rc": proc.returncode}
        # assume R prints JSON to stdout
        try:
            out = json.loads(stdout) if stdout else {}
        except Exception:
            out = {"raw_stdout": stdout}
        return {"error": False, "result": out, "stderr": stderr}
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
