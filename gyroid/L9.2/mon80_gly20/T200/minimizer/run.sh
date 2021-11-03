#!/bin/bash


export GMX_MAXBACKUP=-1

cp ../initial.gro ./

i=0
cp -f em_scaled_steep.mdp min$i.mdp
sed -i "s/repl_itr/$i/g" min$i.mdp
gmx grompp -f min$i.mdp -c initial.gro -p topol.top -r initial.gro -o min$i -maxwarn 5
gmx mdrun -deffnm min$i


for i in `seq 1 8`
do
    j=$((i-1))
    cp em_scaled_steep.mdp min$i.mdp
    sed -i "s/repl_itr/$i/g" min$i.mdp
    gmx grompp -f min$i.mdp -c min$j.gro -p topol.top -r  initial.gro -o min$i -maxwarn 5
    gmx mdrun -v --deffnm min$i
done

mkdir intermidiates
mv min?.* intermidiates

