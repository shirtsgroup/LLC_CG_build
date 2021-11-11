#!/home/subin/pkgs/anaconda3/bin/python

import argparse
from LLC_Membranes.llclib import topology
import numpy as np
import mdtraj
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def initialize():

    parser = argparse.ArgumentParser(description='Fit the parameters of gyroid from the coordinates of the unit cell')
    parser.add_argument('-r', '--ref', default=['I4'], nargs='+',  help='reference atoms')
    parser.add_argument('-g', '--gro', default='scaled_1.0000.gro', type=str, help='gro file')

    return parser

def SchwarzD(x, period):
    """
    :param x: a vector of coordinates (x1, x2, x3)
    :param period: length of one period

    :return: An approximation of the Schwarz D "Diamond" infinite periodic minimal surface
    """

    n = 2*np.pi / period  # might be just pi / period

    a = np.sin(n*x[:, 0])*np.sin(n*x[:, 1])*np.sin(n*x[:, 2])
    b = np.sin(n*x[:, 0])*np.cos(n*x[:, 1])*np.cos(n*x[:, 2])
    c = np.cos(n*x[:, 0])*np.sin(n*x[:, 1])*np.cos(n*x[:, 2])
    d = np.cos(n*x[:, 0])*np.cos(n*x[:, 1])*np.sin(n*x[:, 2])

    return a + b + c + d 

def gyroid(x, period):

    n = 2 * np.pi / period
    a = np.sin(n * x[:, 0]) * np.cos(n * x[:, 1])
    b = np.sin(n * x[:, 1]) * np.cos(n * x[:, 2])
    c = np.sin(n * x[:, 2]) * np.cos(n * x[:, 0])

    return a + b + c

class LLC_structure():

    def __init__(self, gro):

        self.t = mdtraj.load(gro)
        self.positions = self.t.xyz[:, :, :]


    def get_peiod(self):

        return self.t.unitcell_vectors

    def get_atom_pos(self, ref_atoms):

        pore = {}
        pore_positions = {}
        for r in ref_atoms:
            pore[r] = [a.index for a in self.t.topology.atoms if a.name == r]
            pore_positions[r] = self.positions[:, pore[r], :]

        return np.average(list(pore_positions.values()), axis=0)


if __name__ == '__main__':

    args = initialize().parse_args()

    S = LLC_structure(args.gro)
    period = S.get_peiod()[0][0][0]
    ref = args.ref
    print(ref)
    pore = S.get_atom_pos(ref)
    nt, nr, nc = np.shape(pore)


    c_s = np.zeros(nt)
    c_g = np.zeros(nt)
    e_s = np.zeros(nt)
    e_g = np.zeros(nt)

    for i in range(nt):
        c_g[i] = np.average(gyroid(pore[i],  period))
        e_g[i] = np.std(gyroid(pore[i],  period))
        c_s[i] = np.average(SchwarzD(pore[i], period))
        e_s[i] = np.std(SchwarzD(pore[i],  period))

    #print(c_g, e_g)
    print(c_s, e_s)




