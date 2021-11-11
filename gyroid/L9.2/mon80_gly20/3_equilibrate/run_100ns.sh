#!/bin/bash
source /usr/local/gromacs/2020.3_gpu/bin/GMXRC

export GMX_MAXBACKUP=-1
export GMX_MAXCONSTRWARN=-1


# 100 ns long npt
gmx_mpi grompp -f npt_100ns.mdp -p topol_0.top -c npt_long.gro -r npt_long.gro  -o npt_100ns
gmx_mpi  mdrun -v -deffnm npt_1000ns
