"""
Find the Python interpreter in the TDM Studio VM that can actually run the
pipeline, and say exactly what to do about it.

Written for the "no module named openai" dead end. Guessing the conda env name
does not work: the `sample-*` env carries a version suffix that differs per
workbench, ProQuest changes it between images, and the packages are not always
where the docs say. So instead of guessing, this ASKS EVERY interpreter on the
box which packages it has.

The strongest lead it follows: ProQuest ships a working GPT sample notebook
(GPT_Batch_Processing.ipynb). That notebook imports `openai` and runs, so SOME
interpreter on this machine has the SDK -- and Jupyter's kernelspecs record
exactly which one. A kernel whose python has openai is the answer, whatever the
env is called.

Deliberately stdlib-only and old-Python-safe, because it has to run on the very
interpreter that is missing packages. Read-only: it never installs or changes
anything, it only tells you the command to run.

Usage (in the VM Jupyter terminal, from election_arm/ -- any python will do):
    python vm_doctor.py
"""

import glob
import json
import os
import subprocess
import sys

NEEDED = ["openai", "lxml", "pandas"]     # extract_gpt / tdm_parse / analyze
PROBE_TIMEOUT = 25

SEARCH_GLOBS = [
    "/home/ec2-user/SageMaker/.conda/envs/*/bin/python",
    "/home/ec2-user/.conda/envs/*/bin/python",
    "/opt/conda/envs/*/bin/python",
    "/opt/conda/bin/python",
    "/usr/bin/python3",
    "/usr/local/bin/python3",
]
KERNEL_GLOBS = [
    "/home/ec2-user/SageMaker/.conda/envs/*/share/jupyter/kernels/*/kernel.json",
    "/opt/conda/share/jupyter/kernels/*/kernel.json",
    "/usr/local/share/jupyter/kernels/*/kernel.json",
    "/usr/share/jupyter/kernels/*/kernel.json",
    os.path.expanduser("~/.local/share/jupyter/kernels/*/kernel.json"),
]


def probe(python):
    """Ask one interpreter which of NEEDED it can import. (version, {pkg: ver})."""
    code = (
        "import sys, json\n"
        "out = {'py': '%d.%d.%d' % sys.version_info[:3]}\n"
        "for m in " + repr(NEEDED) + ":\n"
        "    try:\n"
        "        mod = __import__(m)\n"
        "        out[m] = getattr(mod, '__version__', 'yes')\n"
        "    except Exception:\n"
        "        out[m] = None\n"
        "print(json.dumps(out))\n"
    )
    try:
        r = subprocess.run([python, "-c", code], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=PROBE_TIMEOUT)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout.decode("utf-8", "replace").strip().splitlines()[-1])
    except Exception:
        return None


def kernel_pythons():
    """Interpreters registered as Jupyter kernels.

    This is the high-value list: ProQuest's own GPT sample notebook runs on one
    of these, and that notebook imports openai -- so if any interpreter on the
    box has the SDK, a kernel python almost certainly does."""
    found = {}
    for pat in KERNEL_GLOBS:
        for path in glob.glob(pat):
            try:
                with open(path) as f:
                    spec = json.load(f)
            except Exception:
                continue
            argv = spec.get("argv") or []
            if argv and isinstance(argv[0], str) and "python" in argv[0]:
                found.setdefault(argv[0], spec.get("display_name", "?"))
    return found


def candidates():
    seen, out = set(), []

    def add(p, why):
        if not p:
            return
        p = os.path.realpath(p)
        if p in seen or not os.path.exists(p):
            return
        seen.add(p)
        out.append((p, why))

    add(sys.executable, "the python running this script")
    for path, name in kernel_pythons().items():
        add(path, "jupyter kernel: %s" % name)
    for pat in SEARCH_GLOBS:
        for p in sorted(glob.glob(pat)):
            add(p, "conda env / system")
    for d in os.environ.get("PATH", "").split(os.pathsep):
        for exe in ("python3", "python"):
            add(os.path.join(d, exe), "on PATH")
    return out


def proquest_artifacts():
    """The other half of the setup: the proxy sample export and its key file."""
    print("\n--- ProQuest proxy artifacts (extract_gpt.py needs these) ---")
    sample = "gpt_sample.txt"
    if os.path.exists(sample):
        try:
            with open(sample, errors="ignore") as f:
                text = f.read()
        except Exception:
            text = ""
        import re
        base = re.search(r'base_url\s*=\s*["\']([^"\']+)["\']', text)
        opens = re.findall(r'open\(\s*["\']([^"\']+)["\']', text)
        key = None
        for p in opens:
            if re.search(r"key|token|cred|secret", p, re.I):
                key = p
                break
        if key is None and opens:
            key = opens[0]
        print("  gpt_sample.txt      FOUND")
        print("    base_url          %s" % (base.group(1) if base else
                                            "NOT FOUND -- re-export the notebook"))
        print("    key file          %s%s" % (
            key or "NOT FOUND",
            "" if (key and os.path.exists(key)) else "   <-- does not exist!"
            if key else ""))
    else:
        print("  gpt_sample.txt      MISSING. Create it with:")
        print("    jupyter nbconvert --to script --stdout \\")
        print("      \".../ProQuest TDM Studio Samples/GPT_Batch_Processing.ipynb\" \\")
        print("      > gpt_sample.txt")


def main():
    print("=" * 72)
    print("TDM Studio VM doctor -- which python can run this pipeline?")
    print("=" * 72)
    print("probing for: %s" % ", ".join(NEEDED))

    rows = []
    for path, why in candidates():
        info = probe(path)
        if info is None:
            continue
        rows.append((path, why, info))

    if not rows:
        raise SystemExit("\nNo working interpreter found at all. That is very "
                         "unexpected -- check `which -a python3`.")

    # Full paths on their own line: conda env paths are long, and truncating
    # them hides the very part (the env name) you need to read.
    print("")
    for path, why, info in rows:
        ok = all(info.get(n) for n in NEEDED)
        print("%s %s" % ("[OK ]" if ok else "[   ]", path))
        print("       python %-8s %s" % (
            info.get("py", "?"),
            "  ".join("%s=%s" % (n, info.get(n) or "MISSING") for n in NEEDED)))
        print("       via %s" % why)

    # Rank: everything importable wins; then openai alone; then nothing.
    def score(r):
        info = r[2]
        return (sum(1 for n in NEEDED if info.get(n)), bool(info.get("openai")))

    best = sorted(rows, key=score, reverse=True)[0]
    have_openai = [r for r in rows if r[2].get("openai")]

    print("\n" + "=" * 72)
    if not have_openai:
        print("VERDICT: NO interpreter on this box has `openai`.")
        print("=" * 72)
        print("""
That is worth knowing -- it means the previous runs used something that is
gone, or this workbench was rebuilt. pip is blocked (no internet), so install
from ProQuest's internal conda mirror into the env you intend to use:

    conda install -n <env-name> openai -y

Find <env-name> with `conda env list` (take the sample-* env that is NOT -r).
If `conda` is not on PATH first run:
    source /home/ec2-user/SageMaker/.conda/etc/profile.d/conda.sh

Then re-run this script to confirm.""")
        return

    path, why, info = best
    missing = [n for n in NEEDED if not info.get(n)]
    print("VERDICT: use this interpreter")
    print("=" * 72)
    print("\n    export PY=%s\n" % path)
    print("  found via: %s" % why)
    print("  python %s | %s" % (info.get("py"), ", ".join(
        "%s %s" % (n, info.get(n) or "MISSING") for n in NEEDED)))

    if missing:
        env = path.split("/envs/")[1].split("/")[0] if "/envs/" in path else "<env>"
        print("\n  NOTE: still missing %s. Install into that env (pip is blocked):"
              % ", ".join(missing))
        print("    conda install -n %s %s -y" % (env, " ".join(missing)))

    print("\n  Then, so the batch script uses the same one:")
    print("    export PY=%s" % path)
    print("\n  And verify end to end:")
    print("    $PY -c \"import extract_gpt, strip_for_export as s; "
          "print(extract_gpt.SCHEMA_VERSION, 'quote' in s.DROP_FIELDS)\"")

    proquest_artifacts()


if __name__ == "__main__":
    main()
