#!/bin/bash

packmol < bilayer.inp
grep ITW bilayer_0.pdb > mon.pdb
grep BR bilayer_0.pdb > ion.pdb
grep GLY bilayer_0.pdb > solv.pdb

grep -v ATOM bilayer_0.pdb | grep -v END >  header.tmp

echo  "CRYST1   60.000   60.000  100.000 90.00  90.00  90.00 P           1" >> header.tmp
cat header.tmp mon.pdb ion.pdb solv.pdb > bilayer.pdb
echo "END" >> bilayer.pdb

mkdir -p trash
mv mon.pdb trash
mv ion.pdb trash
mv solv.pdb trash
mv header.tmp trash








