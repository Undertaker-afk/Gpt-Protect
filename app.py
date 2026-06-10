"""
app.py — HuggingFace Space bootstrapper
======================================
Copy THIS file (alone) into a HuggingFace Space (Gradio SDK). On boot it:

  1. clones / pulls the GPT-Protect GitHub repo,
  2. installs its requirements,
  3. runs the repo's `main.py`, which hosts the Gradio UI and the realtime
     training loop.

Persistent data lives in the Space's mounted bucket at /data, so a restart
instantly resumes training and keeps every collected sample.

Space settings to use:
  * SDK: Gradio
  * app_file: app.py
  * Persistent storage: enabled (mounted at /data)
Optional Space "Variables/Secrets":
  * REPO_URL     (default https://github.com/Undertaker-afk/Gpt-Protect)
  * REPO_BRANCH  (default: repo default branch)
  * MODEL_PRESET (default: tiny  — fits free CPU+16GB; use 0.4b/5b on big HW)
  * DATA_DIR     (default: /data)
"""

import os
import subprocess
import sys

REPO_URL = os.environ.get("REPO_URL", "https://github.com/Undertaker-afk/Gpt-Protect")
REPO_BRANCH = os.environ.get("REPO_BRANCH", "").strip()
# clone into persistent storage when available so re-clone is cheap on restart
_DATA = os.environ.get("DATA_DIR", "/data")
_BASE = _DATA if os.path.isdir(_DATA) and os.access(_DATA, os.W_OK) else os.getcwd()
REPO_DIR = os.environ.get("REPO_DIR", os.path.join(_BASE, "gpt-protect-repo"))


def run(cmd, **kw):
    print(f"[app] $ {cmd}")
    return subprocess.run(cmd, shell=True, check=False, **kw)


def ensure_repo():
    git_dir = os.path.join(REPO_DIR, ".git")
    if os.path.isdir(git_dir):
        print(f"[app] updating existing repo at {REPO_DIR}")
        run(f"git -C {REPO_DIR} fetch --depth 1 origin", )
        run(f"git -C {REPO_DIR} reset --hard "
            f"{'origin/' + REPO_BRANCH if REPO_BRANCH else 'HEAD'}")
        run(f"git -C {REPO_DIR} pull --ff-only")
    else:
        print(f"[app] cloning {REPO_URL} -> {REPO_DIR}")
        branch = f"-b {REPO_BRANCH} " if REPO_BRANCH else ""
        r = run(f"git clone --depth 1 {branch}{REPO_URL} {REPO_DIR}")
        if r.returncode != 0:
            # retry without branch flag
            run(f"git clone --depth 1 {REPO_URL} {REPO_DIR}")


def install_requirements():
    req = os.path.join(REPO_DIR, "requirements.txt")
    if os.path.exists(req):
        print("[app] installing repo requirements …")
        run(f"{sys.executable} -m pip install --no-input -r {req}")
    else:
        print("[app] no requirements.txt in repo; installing minimal deps")
        run(f"{sys.executable} -m pip install --no-input "
            f"torch --index-url https://download.pytorch.org/whl/cpu")
        run(f"{sys.executable} -m pip install --no-input "
            f"transformers datasets tokenizers gradio scikit-learn pandas")


def launch_main():
    main_py = os.path.join(REPO_DIR, "main.py")
    if not os.path.exists(main_py):
        raise SystemExit(f"[app] main.py not found in repo at {main_py}")
    sys.path.insert(0, REPO_DIR)
    os.chdir(REPO_DIR)
    print(f"[app] launching {main_py}")
    import runpy
    runpy.run_path(main_py, run_name="__main__")


def main():
    try:
        ensure_repo()
        install_requirements()
    except Exception as e:
        print(f"[app] setup warning: {e}")
    # If the clone failed but main.py exists next to app.py, fall back to it.
    if not os.path.exists(os.path.join(REPO_DIR, "main.py")) and \
       os.path.exists(os.path.join(os.getcwd(), "main.py")):
        print("[app] falling back to local main.py")
        import runpy
        runpy.run_path("main.py", run_name="__main__")
        return
    launch_main()


if __name__ == "__main__":
    main()
