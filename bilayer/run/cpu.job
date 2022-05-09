#!/bin/bash
#SBATCH -N 1 --ntasks-per-node 16
#SBATCH -t 8:00:00
#SBATCH -p RM-shared
#SBATCH -J 'GI_5'

module load gcc
source /jet/home/susa/pkgs/gromacs/2020.5/bin/GMXRC

export GMX_MAXBACKUP=-1  # do not make back-ups
export GMX_MAXCONSTRWARN=-1

cp ../prepare/bilayer_prod.gro ./

i=0
if [ ! -f prod_$i.xtc ]; then
    gmx grompp -f prod.mdp -p itw_gly.top -c bilayer_prod.gro -o prod_$i
    gmx mdrun -v -deffnm prod_$i -nt 16 -ntomp 1
elif [ ! -f prod_$i.gro ]; then
    gmx mdrun -v -deffnm prod_$i -cpi prod_$i.cpt -nt 16 -ntomp 1
fi

i=1
j=0
if [ ! -f prod_$i.xtc ]; then
    gmx grompp -f prod.mdp -p itw_gly.top -c prod_$j.gro -o prod_$i
    gmx mdrun -v -deffnm prod_$i -nt 16 -ntomp 1
elif [ ! -f prod_$i.gro ]; then
    gmx mdrun -v -deffnm prod_$i -cpi prod_$i.cpt -nt 16 -ntomp 1
fi




