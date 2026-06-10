"""
app.py — HuggingFace Space bootstrapper + auto-updater
=====================================================
Copy THIS file (alone) into a HuggingFace Space (Gradio SDK). On boot it:

  1. clones / pulls the GPT-Protect GitHub repo into persistent storage,
  2. hands off to the repo's copy of app.py (so the supervisor itself stays
     up to date),
  3. installs requirements (only when they change),
  4. launches the repo's `main.py` (Gradio UI + realtime training) as a child
     process, and supervises it.

Auto-update loop (every UPDATE_INTERVAL_SEC, default 20 min):
  * `git fetch` and compare local HEAD vs remote HEAD;
  * if — and only if — they differ, send the child a graceful SIGTERM so it
    **pauses training and writes a checkpoint** to /data, `git pull`, then
    re-exec this supervisor so new code (app.py and/or main.py) takes effect.
    Because all state lives in /data, training resumes from the checkpoint.

Space settings:
  * SDK: Gradio · app_file: app.py · Persistent storage enabled (/data)
Optional Space variables:
  * REPO_URL (default Undertaker-afk/Gpt-Protect) · REPO_BRANCH
  * UPDATE_INTERVAL_SEC (default 1200) · UPDATE_GRACE_SEC (default 150)
  * MODEL_PRESET (default tiny) · DATA_DIR (default /data)
"""

import hashlib
import os
import signal
import subprocess
import sys
import time

REPO_URL = os.environ.get("REPO_URL", "https://github.com/Undertaker-afk/Gpt-Protect")
REPO_BRANCH = os.environ.get("REPO_BRANCH", "").strip()
_DATA = os.environ.get("DATA_DIR", "/data")
_BASE = _DATA if os.path.isdir(_DATA) and os.access(_DATA, os.W_OK) else os.getcwd()
REPO_DIR = os.environ.get("REPO_DIR", os.path.join(_BASE, "gpt-protect-repo"))

UPDATE_INTERVAL = int(os.environ.get("UPDATE_INTERVAL_SEC", "1200"))   # 20 min
UPDATE_GRACE = int(os.environ.get("UPDATE_GRACE_SEC", "150"))
SUPERVISE_TICK = int(os.environ.get("SUPERVISE_TICK_SEC", "30"))
REQS_HASH_FILE = os.path.join(_BASE, ".gptp_reqs_hash")
STATUS_FILE = os.path.join(_DATA if os.path.isdir(_DATA) else _BASE, "updater.json")


def log(msg):
    print(f"[app] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def run(cmd):
    log(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=False)


def git(args):
    return subprocess.run(["git", "-C", REPO_DIR] + args,
                          capture_output=True, text=True)


def _write_status(**kw):
    try:
        import json
        kw["t"] = time.time()
        with open(STATUS_FILE, "w") as f:
            json.dump(kw, f)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
#  repo management
# --------------------------------------------------------------------------- #
def ensure_repo():
    if os.path.isdir(os.path.join(REPO_DIR, ".git")):
        log(f"updating existing repo at {REPO_DIR}")
        git(["fetch", "--depth", "1", "origin"])
        ref = f"origin/{REPO_BRANCH}" if REPO_BRANCH else "FETCH_HEAD"
        git(["reset", "--hard", ref])
    else:
        log(f"cloning {REPO_URL} -> {REPO_DIR}")
        branch = ["-b", REPO_BRANCH] if REPO_BRANCH else []
        r = subprocess.run(["git", "clone", "--depth", "1"] + branch +
                           [REPO_URL, REPO_DIR], capture_output=True, text=True)
        if r.returncode != 0:
            log(f"clone with branch failed ({r.stderr.strip()[:120]}); retrying")
            subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])


def local_remote_diff():
    """(local_sha, remote_sha, differs) — fetches remote first."""
    git(["fetch", "--depth", "1", "origin"])
    local = git(["rev-parse", "HEAD"]).stdout.strip()
    ref = f"origin/{REPO_BRANCH}" if REPO_BRANCH else "FETCH_HEAD"
    remote = git(["rev-parse", ref]).stdout.strip()
    differs = bool(local) and bool(remote) and local != remote
    return local, remote, differs


def git_pull():
    ref = f"origin/{REPO_BRANCH}" if REPO_BRANCH else "FETCH_HEAD"
    git(["fetch", "--depth", "1", "origin"])
    git(["reset", "--hard", ref])


def install_requirements(force=False):
    req = os.path.join(REPO_DIR, "requirements.txt")
    if not os.path.exists(req):
        return
    h = hashlib.md5(open(req, "rb").read()).hexdigest()
    prev = ""
    if os.path.exists(REQS_HASH_FILE):
        prev = open(REQS_HASH_FILE).read().strip()
    if force or h != prev:
        log("installing requirements (changed)…")
        run(f"{sys.executable} -m pip install --no-input -r {req}")
        try:
            open(REQS_HASH_FILE, "w").write(h)
        except Exception:
            pass
    else:
        log("requirements unchanged; skipping install")


# --------------------------------------------------------------------------- #
#  child process supervision
# --------------------------------------------------------------------------- #
def launch_main():
    main_py = os.path.join(REPO_DIR, "main.py")
    if not os.path.exists(main_py):
        raise SystemExit(f"[app] main.py not found at {main_py}")
    log("launching main.py")
    env = dict(os.environ)
    return subprocess.Popen([sys.executable, "-u", "main.py"],
                            cwd=REPO_DIR, env=env)


def graceful_stop(proc):
    """Ask main.py to checkpoint + exit; escalate to kill if it overruns."""
    if proc.poll() is not None:
        return
    log(f"SIGTERM -> main.py (grace {UPDATE_GRACE}s for checkpoint)")
    try:
        proc.send_signal(signal.SIGTERM)
    except Exception:
        pass
    t0 = time.time()
    while time.time() - t0 < UPDATE_GRACE:
        if proc.poll() is not None:
            log("main.py exited cleanly (checkpoint saved)")
            return
        time.sleep(1.0)
    log("grace expired; SIGKILL")
    try:
        proc.kill()
    except Exception:
        pass


def supervise():
    proc = launch_main()
    last_check = time.time()
    local0 = git(["rev-parse", "HEAD"]).stdout.strip()
    _write_status(local_sha=local0, remote_sha=local0, updating=False,
                  last_check=time.time(), interval=UPDATE_INTERVAL)
    while True:
        time.sleep(SUPERVISE_TICK)

        # crash recovery
        if proc.poll() is not None:
            log(f"main.py exited rc={proc.returncode}; relaunching")
            proc = launch_main()

        # periodic update check
        if time.time() - last_check >= UPDATE_INTERVAL:
            last_check = time.time()
            try:
                local, remote, differs = local_remote_diff()
            except Exception as e:
                log(f"update check failed: {e}")
                continue
            _write_status(local_sha=local, remote_sha=remote, updating=differs,
                          last_check=time.time(), interval=UPDATE_INTERVAL)
            if not differs:
                log(f"no update (HEAD {local[:7]})")
                continue
            log(f"update available {local[:7]} -> {remote[:7]}")
            graceful_stop(proc)
            git_pull()
            install_requirements()
            log("re-exec supervisor with updated code")
            _write_status(local_sha=remote, remote_sha=remote, updating=False,
                          last_check=time.time(), last_update=time.time(),
                          interval=UPDATE_INTERVAL)
            os.execv(sys.executable,
                     [sys.executable, os.path.abspath(__file__)] + sys.argv[1:])


# --------------------------------------------------------------------------- #
def main():
    try:
        ensure_repo()
    except Exception as e:
        log(f"ensure_repo warning: {e}")

    # Hand off to the repo's own app.py so the supervisor is itself updatable.
    repo_app = os.path.join(REPO_DIR, "app.py")
    if (os.environ.get("GPTP_FROM_REPO") != "1"
            and os.path.exists(repo_app)
            and os.path.abspath(repo_app) != os.path.abspath(__file__)):
        os.environ["GPTP_FROM_REPO"] = "1"
        os.chdir(REPO_DIR)
        log("handoff -> repo app.py")
        os.execv(sys.executable, [sys.executable, repo_app] + sys.argv[1:])

    install_requirements()

    # Fallback: if main.py somehow isn't in the repo, run a local one.
    if not os.path.exists(os.path.join(REPO_DIR, "main.py")) and \
       os.path.exists(os.path.join(os.getcwd(), "main.py")):
        log("repo main.py missing; running local main.py")
        import runpy
        runpy.run_path("main.py", run_name="__main__")
        return

    supervise()


if __name__ == "__main__":
    main()
