#!/usr/bin/env python3
"""Fetch (or explain how to fetch) the official fly-brain resources.

This repository does not bundle the FlyWire connectome or the official
Shiu et al. 2024 whole-brain model. Both are large.

Official code (MIT license), as published with Shiu, P.K. et al.,
"A Drosophila computational brain model reveals sensorimotor processing",
Nature 634:210–219 (2024):

    https://github.com/philshiu/Drosophila_brain_model

The FlyWire adult fly brain connectome (CC BY-NC 4.0) is described in
Dorkenwald et al., Nature 634:124–138 (2024). Connectivity files used
by the Shiu model ship with that repository (v630 by default; v783 as
an option).

A neuromusculoskeletal body, NeuroMechFly / FlyGym, is maintained by
the Ramdya lab and is a *separate*, optional coupling. The published
Shiu model is a brain-only LIF network: it has no body, no learning,
and simplified synapses. Treat full embodiment as a stretch goal.

This script clones the official model repository into data/flywire/ if
git and the network are available. It never commits those files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/philshiu/Drosophila_brain_model.git"


def main() -> int:
    dest = Path("data/flywire/Drosophila_brain_model")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"already present: {dest}")
        return 0
    print(f"cloning {REPO} into {dest} ...")
    try:
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(dest)], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        print("Automatic download failed.", file=sys.stderr)
        print(f"Please clone manually:\n  git clone --depth 1 {REPO} {dest}", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        return 1
    print("done. The official resources are not part of this git repository.")
    print("A tiny synthetic connectome for offline tests is in tests/fixtures/tiny_connectome.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
