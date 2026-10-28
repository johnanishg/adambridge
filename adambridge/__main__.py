"""``python -m adambridge {build,info}``: build the native kernels or show what will be used."""

import sys

from .native import _main

if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
