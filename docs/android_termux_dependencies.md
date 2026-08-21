# Android / Termux dependencies

SharpEdge has two dependency lanes:

- Runtime analytics dependencies: `requirements.txt`
- Developer tooling: `requirements-dev.txt`

## Recommended Termux install

NumPy on Android/Termux should come from Termux packages, not PyPI. PyPI may try
to compile NumPy from source and fail on Android libc/math edge cases.

```bash
pkg update
pkg install python python-numpy ruff
python -m pip install -r requirements.txt
```

Do **not** install `requirements-dev.txt` with pip on Termux unless you know a
wheel exists. Ruff is already provided by `pkg`; pip may try to build Ruff from
source and waste your afternoon like a tiny Rust-powered space heater.

If the active Python is a virtual environment, create it with system packages so
it can see Termux's compiled NumPy:

```bash
python -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
```

Existing venvs can be checked with:

```bash
cat .venv/pyvenv.cfg
```

Look for:

```text
include-system-site-packages = true
```

If it is `false`, the venv cannot see `pkg install python-numpy`.

## Ruff

Termux packages Ruff directly:

```bash
pkg install ruff
ruff --version
```

Use the `ruff` command directly. `python -m ruff` may fail in a venv because the
Termux package installs a binary, not necessarily a Python module inside that
venv.

The repo also tracks Ruff in `requirements-dev.txt` for non-Android or wheel-capable
environments.

## Pandas note

`requirements.txt` includes pandas because several analytics scripts still use it.
On Android/Termux, pandas may need either a compatible wheel or a Termux package
from the active repository set. If pip attempts a source build and fails, prefer
using a prebuilt Termux package/repository or run pandas-heavy analytics from a
Linux/desktop environment.
