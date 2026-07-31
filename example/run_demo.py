#!/usr/bin/env python
"""Runnable demo: predict the TB-mBJ c* for SnS2 from the bundled cheap-PBE inputs."""
import os, sys, subprocess
here = os.path.dirname(os.path.abspath(__file__))
cmd = [sys.executable, os.path.join(here, "..", "src", "predict_cstar.py"),
       "--contcar",     os.path.join(here, "SnS2", "CONTCAR"),
       "--chgcar",      os.path.join(here, "SnS2", "CHGCAR.gz"),
       "--dos-vasprun", os.path.join(here, "SnS2", "dos_vasprun.xml.gz"),
       "--pbe-gap",     "1.56"]
sys.exit(subprocess.call(cmd))
