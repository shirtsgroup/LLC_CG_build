Building new polymers
=====================

AA parameterization
-------------------
Use the Jupyter create_monomer.ipynb to create the all-atom
parameterization.  It requires several packages to be installed (which
are described at the top of the script).

You will need to reorganize the output in the .top files to make itp's
for individual molecules.  SEE AA/FSI_AA.itp, AA/solvent_AA.itp,
AA/ions_AA.itp. and compare to AA/monomer.top and AA/solvent.top to see how
the information is reorganized.

CG parameterization
-------------------
Once you have the AA parameterization, then use this tool 

https://jbarnoud.github.io/cgbuilder/

to build the CC mode. You will need to use the output.gro from the AA
step.  It will produce the CG files (.ndx,.map,.gro) using the output
pdb from the md simulation (see exanples).

You will have to build the CG.itp by hand.  See the example .itps the
CC/ directory.

To decide on the bead types, use the rules for assigning atom types
laid out here:
https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-021-01098-3/MediaObjects/41592_2021_1098_MOESM1_ESM.pdf

The bond/angle/dihedral parameters are stubs. To decide what bonds
lengths and strengths to use, see instructions here.

http://cgmartini.nl/index.php/martini-3-tutorials/parameterizing-a-new-small-molecule