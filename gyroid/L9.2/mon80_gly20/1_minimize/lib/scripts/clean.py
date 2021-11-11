#!/usr/bin/env python
import os
import subprocess


if not os.path.isdir("./intermediates"):
    os.mkdir('intermediates')

if not os.path.isdir("./trash"):
    os.mkdir('trash')

# CLEAN UP -- move all scaled unit cells into a separate directory
mv = "mv scaled*.gro intermediates"
p = subprocess.Popen(mv, shell=True)  # don't split mv because shell=True
p.wait()


mv = "mv solvated_nvt* solvated_npt* npt_equil* npt_* nvt_equil* em_* intermediates"
p = subprocess.Popen(mv, shell=True)  # don't split mv because shell=True
p.wait()

mv = "mv step*.pdb trash"
p = subprocess.Popen(mv, shell=True)
p.wait()


