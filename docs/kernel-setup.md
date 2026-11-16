# Why the kernel needs installing, and what happens if you skip it

## The failure

A notebook started with the stock `python3` kernelspec, or with the interpreter picked
directly out of the conda environment, dies on the first cell. The front end reports
only *"the kernel died"* — no traceback, no Python error.

Under `-X faulthandler` the abort is `0xc06d007f`, a delay-load dependency failure,
inside NumPy's `blas_fpe_check` at import. The process exit code is `-1066598273`.

## The cause

NumPy delay-loads its BLAS library (`libblas.dll`) from the environment's
`Library\bin`. Windows resolves a delay-loaded DLL through **PATH**. Launching
`<env>\python.exe` without activating the environment leaves `Library\bin` off PATH,
so the load fails as a structured exception rather than a Python `ImportError` — which
aborts the process instead of raising.

`zmq` and `ipykernel` import fine, which is why the kernel appears to start and only
then dies.

## The two fixes, both applied by `tools/install_kernel.py`

1. **A per-user kernelspec that activates the environment.** Its `argv` points at a
   generated `.bat` that runs `call "<root>\Scripts\activate.bat" bioproc` and then
   `python -m ipykernel_launcher`. The activation must live in a batch file, not in
   `argv`: a kernelspec's `argv` is handed to `CreateProcess`, which re-quotes each
   element, so a quoted path arrives at `cmd.exe` escaped as `\"` and fails. In VS Code
   this appears under *Jupyter Kernel* as **Python (bioproc)**.

2. **A `.pth` in the environment's `site-packages`** that prepends `Library\bin` to
   `os.environ["PATH"]` at interpreter startup, before any import. This covers front
   ends that bypass the kernelspec and launch the interpreter themselves — which is
   what VS Code does when the kernel is chosen under *Python Environments* (shown
   there as `bioproc (Python 3.11)`).

`os.add_dll_directory` does **not** fix this: it affects `LoadLibraryEx` searches,
while delay-loading goes through PATH. Setting `os.environ["PATH"]` does work.

## When to re-run it

The kernelspec lives outside the environment and survives rebuilding its packages. The
`.pth` lives inside it and does not — **re-run `python tools/install_kernel.py` after
recreating the environment**.
