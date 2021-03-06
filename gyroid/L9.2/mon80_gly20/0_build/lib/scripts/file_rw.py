#!/usr/bin/env python

"""
This library has all routines involving reading and writing files
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import copy
import math
import os
import pickle


def read_pdb_coords(file):

    a = []
    for line in file:
        a.append(line)
    file.close()

    no_atoms = 0  # number of atoms in one monomer including sodium ion
    for i in range(0, len(a)):
        no_atoms += a[i].count('ATOM')

    lines_of_text = 0  # lines of text at top of .pdb input file
    for i in range(0, len(a)):
        if a[i].count('ATOM') == 0:
            lines_of_text += 1
        if a[i].count('ATOM') == 1:
            break

    xyz = np.zeros([3, no_atoms])
    identity = np.zeros([no_atoms], dtype=object)
    for i in range(lines_of_text, lines_of_text + no_atoms):  # searches relevant lines of text in file, f, being read
        xyz[:, i - lines_of_text] = [float(a[i][26:38]), float(a[i][38:46]), float(a[i][46:54])]  # Use this to read specific entries in a text file
        identity[i - lines_of_text] = str.strip(a[i][12:16])

    return xyz, identity, no_atoms, lines_of_text


def read_gro_coords(file):

    a = []
    for line in file:
        a.append(line)
    file.close()

    lines_of_text = 2  # Hard Coded -> BAD .. but I've seen this in mdtraj scripts
    no_atoms = len(a) - lines_of_text - 1  # subtract one for the bottom box vector line

    xyz = np.zeros([3, no_atoms])
    identity = np.zeros([no_atoms], dtype=object)
    for i in range(lines_of_text, lines_of_text + no_atoms):  # searches relevant lines of text in file, f, being read
        xyz[:, i - lines_of_text] = [float(a[i][20:28])*10, float(a[i][28:36])*10, float(a[i][36:44])*10]
        identity[i - lines_of_text] = str.strip(a[i][11:16])

    return xyz, identity, no_atoms, lines_of_text


def write_assembly(b, output, no_mon, xlink=False):
    """
    :param b: Name of build monomer (string)
    :param output: name of output file
    :param no_mon: number of monomers in the assembly
    :param xlink : whether the system is being cross-linked
    :return:
    """
    # print up to ' [ atoms ] ' since everything before it does not need to be modified
    #location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))  # Location of this script
    nres = len(b)  # number of different residues

    section_indices = []
    itps = []
    for m, res in enumerate(b):

        if type(res) is list:  # restrain.py might pass in an already modified topology that isn't in the below folders
            itps.append(b)
        else:
            with open("lib/topologies/%s" % ('%s.itp' % res), "r") as f:
                a = []
                for line in f:
                    a.append(line)
            itps.append(a)

        section_indices.append(get_indices(itps[m], xlink))
        # atoms_index, bonds_index, pairs_index, angles_index, dihedrals_p_index, \
        # dihedrals_imp_index, vsite_index, vtype = get_indices(a, xlink)

    # for i in range(0, atoms_index + 2):  # prints up to and including [ atoms ] in addition to the header line after it
    #     f.write(a[i])

    f = open('%s' % output, 'w')

    f.write('[ moleculetype ]\n')
    f.write(';name           nrexcl\n')
    f.write('restrained         1\n')
    f.write('\n[ atoms ]\n')

    # [ atoms ]
    natoms = []
    start_ndx = 0
    for r in range(nres):

        a = itps[r]
        atoms_index = section_indices[r]['atoms_index']
        atoms_count = atoms_index + 2
        nr = 0  # number of atoms
        while a[atoms_count] != '\n':
            atoms_count += 1  # increments the while loop
            nr += 1  # counts number of atoms

        natoms.append(nr)

        for i in range(int(no_mon[r])):  # print atom information for each monomer
            for k in range(0, nr):  # getting the number right
                f.write('{:5d}{:25s}{:5d}{:}'.format(i*nr + k + 1 + start_ndx, a[k + atoms_index + 2][6:29],
                                                   i*nr + int(a[k + atoms_index + 2][29:34]) + start_ndx,
                                                   a[k + atoms_index + 2][34:len(a[k + atoms_index + 2])]))

        start_ndx = int(no_mon[r]*nr)

    f.write("\n[ bonds ]\n")

    # [ bonds ]

    start_ndx = 0
    for r in range(nres):

        a = itps[r]
        bonds_index = section_indices[r]['bonds_index']

        nb = 0  # number of lines in the 'bonds' section
        bond_count = bonds_index + 2
        while a[bond_count] != '\n':
            bond_count += 1  # increments while loop
            nb += 1  # counting number of lines in 'bonds' section

        nr = natoms[r]

        for i in range(int(no_mon[r])):
            for k in range(0, nb):
                f.write('{:6d}{:7d}{:}'.format(i*nr + int(a[k + bonds_index + 2][0:6]) + start_ndx,
                                               i*nr + int(a[k + bonds_index + 2][6:14]) + start_ndx,
                                               a[k + bonds_index + 2][14:]))

        start_ndx = int(no_mon[r]*nr)

        # [ constraints ]
        f.write("\n[ constraints ]\n")

        start_ndx = 0
        for r in range(nres):

            a = itps[r]
            constraints_index = section_indices[r]['constraints_index']

            nb = 0  # number of lines in the 'constraints' section
            constraint_count = constraints_index + 2
            while a[constraint_count] != '\n':
                constraint_count += 1  # increments while loop
                nb += 1  # counting number of lines in 'constraints' section

            nr = natoms[r]

            for i in range(int(no_mon[r])):
                for k in range(0, nb):
                    f.write('{:6d}{:7d}{:}'.format(i * nr + int(a[k + constraints_index + 2][0:6]) + start_ndx,
                                                   i * nr + int(a[k + constraints_index + 2][6:14]) + start_ndx,
                                                   a[k + constraints_index + 2][14:]))

            start_ndx = int(no_mon[r] * nr)


    # [ pairs ]

    f.write("\n[ pairs ]\n")

    start_ndx = 0
    for r in range(nres):

        a = itps[r]
        pairs_index = section_indices[r]['pairs_index']
        nr = natoms[r]

        npair = 0  # number of lines in the 'pairs' section
        pairs_count = pairs_index + 2  # keep track of index of a
        while a[pairs_count] != '\n':
            pairs_count += 1
            npair += 1

        for i in range(int(no_mon[r])):
            for k in range(0, npair):
                f.write('{:6d}{:7d}{:}'.format(i*nr + int(a[k + pairs_index + 2][0:6]) + start_ndx,
                                               i*nr + int(a[k + pairs_index + 2][6:14]) + start_ndx,
                                               a[k + pairs_index + 2][14:len(a[k + pairs_index + 2])]))

        start_ndx = int(no_mon[r] * nr)

    # [ angles ]

    f.write("\n[ angles ]\n")

    start_ndx = 0
    for r in range(nres):

        a = itps[r]
        angles_index = section_indices[r]['angles_index']

        na = 0  # number of lines in the 'angles' section
        angle_count = angles_index + 2  # keep track of index of a
        while a[angle_count] != '\n':
            angle_count += 1
            na += 1

        nr = natoms[r]

        for i in range(int(no_mon[r])):
            for k in range(0, na):
                f.write('{:6d}{:7d}{:7d}{:}'.format(i*nr + int(a[k + angles_index + 2][0:6]) + start_ndx,
                                                    i*nr + int(a[k + angles_index + 2][6:14]) + start_ndx,
                                                    i*nr + int(a[k + angles_index + 2][14:22]) + start_ndx,
                                                    a[k + angles_index + 2][22:len(a[k + angles_index + 2])]))
        start_ndx = int(no_mon[r] * nr)

    # [ dihedrals ] ; propers

    f.write("\n[ dihedrals ] ; propers\n")  # space in between sections

    start_ndx = 0
    for r in range(nres):

        a = itps[r]
        dihedrals_p_index = section_indices[r]['dihedrals_p_index']

        # TODO: rewrite these so they just ignore all the comment lines
        ndp = 0  # number of lines in the 'dihedrals ; proper' section
        dihedrals_p_count = dihedrals_p_index + 3  # keep track of index of a
        while a[dihedrals_p_count] != '\n':
            dihedrals_p_count += 1
            ndp += 1

        nr = natoms[r]

        for i in range(int(no_mon[r])):
            for k in range(0, ndp):
                #info = [int(x) for x in a[k + dihedrals_p_index + 3].split()[:7]]

                # f.write('{:6d}{:7d}{:7d}{:7d}{:}'.format(i*nr + int(a[k + dihedrals_p_index + 3][0:6]),
                #                                        i*nr + int(a[k + dihedrals_p_index + 3][6:14]),
                #                                        i*nr + int(a[k + dihedrals_p_index + 3][14:22]),
                #                                        i*nr + int(a[k + dihedrals_p_index + 3][22:30]),
                #                                        a[k + dihedrals_p_index + 3][30:len(a[k + dihedrals_p_index + 3])]))
                #f.write('{:6d}{:7d}{:7d}{:7d}{:7d}{:7d}{:7d}\n'.format(i * nr + info[0] + start_ndx, i * nr + info[1] + start_ndx,
                #                                             i * nr + info[2] + start_ndx, i * nr + info[3] + start_ndx,
                #                                             info[4], info[5], info[6]))

                # Subin: Addintional info for bending tortion
                info = [int(x) for x in a[k + dihedrals_p_index + 3].split()[:11]]
                f.write('{:6d}{:7d}{:7d}{:7d}{:7d}{:7d}{:7d}{:7d}{:7d}{:7d}{:7d}\n'.format(i * nr + info[0] + start_ndx,
                                                                       i * nr + info[1] + start_ndx,
                                                                       i * nr + info[2] + start_ndx,
                                                                       i * nr + info[3] + start_ndx,
                                                                       info[4], info[5], info[6], info[7], info[8], info[9], info[10]))

        start_ndx = int(no_mon[r] * nr)

    # [ dihedrals ] ; impropers

    f.write("\n[ dihedrals ] ; impropers\n")  # space in between sections

    start_ndx = 0
    for r in range(nres):

        a = itps[r]
        dihedrals_imp_index = section_indices[r]['dihedrals_imp_index']

        ndimp = 0  # number of lines in the 'dihedrals ; impropers' section
        dihedrals_imp_count = dihedrals_imp_index + 3

        while dihedrals_imp_count < len(a) and a[dihedrals_imp_count] != '\n':
            dihedrals_imp_count += 1
            ndimp += 1

        nr = natoms[r]

        # Can't have any space at the bottom of the file for this loop to work
        for i in range(int(no_mon[r])):
            for k in range(0, ndimp):
                info = [int(x) for x in a[k + dihedrals_imp_index + 3].split()[:7]]
                f.write('{:6d}{:7d}{:7d}{:7d}{:7d}{:7d}{:7d}\n'.format(i * nr + info[0] + start_ndx, i * nr + info[1] + start_ndx,
                                                             i * nr + info[2] + start_ndx, i * nr + info[3] + start_ndx,
                                                             info[4], info[5], info[6]))
        start_ndx = int(no_mon[r] * nr)

    f.write("\n")  # space in between sections

    # [ virtual_sites4 ]
    # NOTE: untested for multiple residues
    start_ndx = 0
    for r in range(nres):

        vsite_index = section_indices[r]['vsite_index']

        if vsite_index is not None:

            f.write("\n[ virtual_sites4 ]\n")

            a = itps[r]

            nv = 0
            vsite_count = vsite_index + 2

            for i in range(vsite_count, len(a)):  # This is the last section in the input .itp file
                vsite_count += 1
                nv += 1

            nr = natoms[r]

            if section_indices[r]['vtype'] == '3fd':
                for i in range(int(no_mon[r])):
                    for k in range(nv):
                        f.write('{:<6d}{:<6d}{:<6d}{:<6d}{:<6d}{:<8.4f}{:<8.4f}\n'.format(
                            i*nr + int(a[k + vsite_index + 1][0:6]) + start_ndx,
                            i*nr + int(a[k + vsite_index + 1][6:12]) + start_ndx,
                            i*nr + int(a[k + vsite_index + 1][12:18]) + start_ndx,
                            i*nr + int(a[k + vsite_index + 1][18:24]) + start_ndx,
                            int(a[k + vsite_index + 1][24:30]), float(a[k + vsite_index + 1][30:38]),
                            float(a[k + vsite_index + 1][38:])))
            elif xlink:

                # Make sure there is no space at the bottom of the topology if you are getting errors
                for i in range(int(no_mon[r])):
                    for k in range(0, nv):
                        f.write('{:<8d}{:<6d}{:<6d}{:<6d}{:<8d}{:<8d}{:<11}{:<11}{:}'.format(
                            i*nr + int(a[k + vsite_index + 2][0:8]) + start_ndx,
                            i*nr + int(a[k + vsite_index + 2][8:14]) + start_ndx,
                            i*nr + int(a[k + vsite_index + 2][14:20]) + start_ndx,
                            i*nr + int(a[k + vsite_index + 2][20:26]) + start_ndx,
                            i*nr + int(a[k + vsite_index + 2][26:34]) + start_ndx,
                            int(a[k + vsite_index + 2][34:42]), a[k + vsite_index + 2][42:53],
                            a[k + vsite_index + 2][53:64], a[k + vsite_index + 2][64:len(a[k + vsite_index + 2])]))
        start_ndx = int(no_mon[r] * nr)

    f.close()


def get_indices(a, xlink):
    # find the indices of all fields that need to be modified
    atoms_index = 0  # find index where [ atoms ] section begins
    while a[atoms_index].count('[ atoms ]') == 0:
        atoms_index += 1

    bonds_index = 0  # find index where [ bonds ] section begins
    while a[bonds_index].count('[ bonds ]') == 0:
        bonds_index += 1

    constraints_index = 0  # find index where [ constraint ] section begins
    while a[constraints_index].count('[ constraints ]') == 0:
        constraints_index += 1

    pairs_index = 0  # find index where [ pairs ] section begins
    while a[pairs_index].count('[ pairs ]') == 0:
        pairs_index += 1

    angles_index = 0  # find index where [ angles ] section begins
    while a[angles_index].count('[ angles ]') == 0:
        angles_index += 1

    dihedrals_p_index = 0  # find index where [ dihedrals ] section begins (propers)
    while a[dihedrals_p_index].count('[ dihedrals ] ; propers') == 0:
        dihedrals_p_index += 1

    dihedrals_imp_index = 0  # find index where [ dihedrals ] section begins (impropers)
    while a[dihedrals_imp_index].count('[ dihedrals ] ; impropers') == 0:
        dihedrals_imp_index += 1

    # if xlink == 'on':
    try:
        vsite_index = 0  # find index where [ dihedrals ] section begins (propers)
        while a[vsite_index].count('[ virtual_sites') == 0:
            vsite_index += 1
        vtype = a[vsite_index].split('virtual_sites')[1].split()[0]
        vfunc = a[vsite_index + 1].split()[4]
        if vtype == '3' and vfunc == '2':
            vtype = vtype + 'fd'
    except IndexError:
        vsite_index = None
        vtype = None
    # else:
    #     vsite_index = 0

    return {'atoms_index': atoms_index, 'bonds_index': bonds_index, 'constraints_index': constraints_index, 'pairs_index': pairs_index,
            'angles_index': angles_index, 'dihedrals_p_index': dihedrals_p_index,
            'dihedrals_imp_index': dihedrals_imp_index, 'vsite_index': vsite_index, 'vtype': vtype}


def write_initial_config(positions, identity, name, no_layers, layer_distribution, dist, no_pores, p2p, no_ions, rot, out,
              offset, helix, offset_angle, *flipped):

    write_gro_pos(positions.T / 10, 'test.gro')

    f = open('%s' % out, 'w')

    f.write('This is a .gro file\n')
    sys_atoms = sum(layer_distribution)*positions.shape[1]
    f.write('%s\n' % sys_atoms)

    rot *= np.pi / 180  # convert input (degrees) to radians

    if flipped:
        flipped = np.asarray(flipped)
        flipped = np.reshape(flipped, positions.shape)
        flip = 'yes'
        unflipped = copy.deepcopy(positions)
    else:
        flip = 'no'

    # main monomer
    atom_count = 1
    monomer_count = 0
    no_atoms = positions.shape[1]
    for l in range(0, no_pores):  # loop to create multiple pores
        # b = grid[0, l]
        # c = grid[1, l]
        theta = 30  # angle which will be used to do hexagonal packing
        if l == 0:  # unmodified coordinates
            b = 0
            c = 0
        elif l == 1:  # move a pore directly down
            b = -1
            c = 0
            if flip == 'yes':
                positions[:, :] = flipped
        elif l == 2:  # moves pore up and to the right
            b = -math.sin(math.radians(theta))
            c = -math.cos(math.radians(theta))
            if flip == 'yes':
                positions[:, :] = unflipped
        elif l == 3:  # moves a pore down and to the right
            b = math.cos(math.radians(90 - theta))
            c = -math.sin(math.radians(90 - theta))
            if flip == 'yes':
                positions[:, :] = flipped
        for k in range(no_layers):
            layer_mons = layer_distribution[l*no_layers + k]
            for j in range(layer_mons):  # iterates over each monomer to create coordinates
                monomer_count += 1
                theta = j * math.pi / (layer_mons / 2.0) + rot
                theta += k * math.pi * (offset_angle / 180)
                if offset:
                    theta += (k % 2) * (math.pi / layer_mons)
                Rx = transform.rotate_z(theta)
                xyz = np.zeros(positions.shape)
                for i in range(positions.shape[1] - no_ions):
                    if helix:
                        xyz[:, i] = np.dot(Rx, positions[:, i]) + [b*p2p, c*p2p, k*dist + (dist / float(layer_mons))*j]
                        hundreds = int(math.floor(atom_count / 100000))
                    else:
                        if k % 2 == 0:
                            xyz[:, i] = np.dot(Rx, positions[:, i]) + [b*p2p, c*p2p, k*dist - 0.5*dist]
                        else:
                            xyz[:, i] = np.dot(Rx, positions[:, i]) + [b*p2p, c*p2p, k*dist]
                        # xyz[:, i] = np.dot(Rx, positions[:, i]) + [b, c, k*dist]
                        hundreds = int(math.floor(atom_count / 100000))
                    f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}'.format(monomer_count, name, identity[i],
                        atom_count - hundreds*100000, xyz[0, i] / 10.0, xyz[1, i] / 10.0, xyz[2, i] / 10.0) + "\n")
                    atom_count += 1

    # Ions:

    for l in range(no_pores):  # loop to create multiple pores
        # b = grid[0, l]
        # c = grid[1, l]
        theta = 30  # angle which will be used to do hexagonal packing
        if l == 0:  # unmodified coordinates
            b = 0
            c = 0
        elif l == 1:  # move a pore directly down
            b = - 1
            c = 0
        elif l == 2:  # moves pore up and to the right
            b = math.cos(math.radians(90 - theta))
            c = -math.sin(math.radians(90 - theta))
        elif l == 3:  # moves a pore down and to the right
            b = -math.sin(math.radians(theta))
            c = -math.cos(math.radians(theta))
        for k in range(no_layers):
            layer_mons = layer_distribution[l*no_layers + k]
            for j in range(layer_mons):  # iterates over each monomer to create coordinates
                theta = j * math.pi / (layer_mons / 2.0) + rot
                if offset:
                    theta += (k % 2) * (math.pi / layer_mons) + rot
                Rx = transform.rotate_z(theta)
                xyz = np.zeros([3, no_ions])
                for i in range(0, no_ions):
                    monomer_count += 1
                    if helix:
                        xyz[:, i] = np.dot(Rx, positions[:, no_atoms - (i + 1)]) + [b*p2p, c*p2p, k*dist + (dist / float(layer_mons))*j]
                        hundreds = int(math.floor(atom_count / 100000))
                    else:
                        xyz[:, i] = np.dot(Rx, positions[:, no_atoms - (i + 1)]) + [b*p2p, c*p2p, k*dist]
                        # xyz[:, i] = np.dot(Rx, positions[:, no_atoms - (i + 1)]) + [b, c, k*dist]
                        hundreds = int(math.floor(atom_count / 100000))
                    f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}'.format(monomer_count, identity[no_atoms - (i + 1)],
                        identity[no_atoms - (i + 1)], atom_count - hundreds*100000, xyz[0, i] / 10.0, xyz[1, i] / 10.0,
                        xyz[2, i] / 10.0) + "\n")
                    atom_count += 1

    f.write('   0.00000   0.00000  0.00000\n')
    f.close()


def last_frame(trr, gro):

    import mdtraj as md

    if trr.endswith('.trr') or trr.endswith('.xtc'):

        t = md.load('%s' % trr, top='%s' % gro)
        last = t.slice(-1)

        pos = t.xyz

        # 'last' will hold all gro information
        res_no = [a.residue.index + 1 for a in t.topology.atoms]
        res_name = [a.residue.name for a in t.topology.atoms]


        last = np.zeros([pos.shape[1], pos.shape[2]])
        last[:, :] = pos[-1, :, :]

    else:
        print('Incompatible Filetype')

    return last


def write_gro(t, out, frame=-1):

    """
    :param t: mdtraj trajectory object. To get a single frame, use t.slice(frame_no)
    :param out: name of gro file to write
    :param frame: frame number to write
    :return: single frame gro file written to disk
    """
    pos = t.xyz
    v = t.unitcell_vectors

    with open(out, 'w') as f:

        f.write('This is a .gro file\n')
        f.write('%s\n' % t.n_atoms)

        count = 0

        d = {'H1': 'HW1', 'H2': 'HW2', 'O': 'OW'}  # mdtraj renames water residues for some unhelpful reason

        for a in t.topology.atoms:
            if a.residue.name == 'HOH':
                f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}\n'.format(a.residue.index + 1, 'SOL', d[a.name],
                                                    count + 1, pos[frame, count, 0], pos[frame, count, 1], pos[frame, count, 2]))
            else:
                f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}\n'.format((a.residue.index + 1) % 100000, a.residue.name, a.name,
                                            (count + 1) % 100000, pos[frame, count, 0], pos[frame, count, 1], pos[frame, count, 2]))
            count += 1

        f.write('{:10f}{:10f}{:10f}{:10f}{:10f}{:10f}{:10f}{:10f}{:10f}\n'.format(v[frame, 0, 0], v[frame, 1, 1], v[frame, 2, 2],
                                                                                  v[frame, 0, 1], v[frame, 2, 0], v[frame, 1, 0],
                                                                                  v[frame, 0, 2], v[frame, 1, 2], v[frame, 2, 0]))


def write_water_ndx(keep, t):
    """ Generate index groups for waters inside membrane. The indices are the same as those in the fully solvated
    structure """

    waters = []
    membrane = []
    for a in t.topology.atoms:
        if a.index in keep and 'HOH' in str(a.residue):  # if the atom is being kept and is part of water, record it
            waters.append(a.index)
        elif a.index in keep:  # otherwise it is part of the membrane. Needs to be in keep though or else the unkept \
            membrane.append(a.index)  # water will go in the membrane list where they aren't supposed to >:(

    count = 1
    with open('water_index.ndx', 'w') as f:  # open up an index file to write to

        f.write('[  water  ]\n')  # first index group
        for index in waters:
            if count % 10 != 0:  # every 10 entries, make a new line
                f.write('{:<8s}'.format(str(index + 1)))  # things are indexed starting at 0 in mdtraj and 1 in gromacs
            else:
                f.write('{:<8s}\n'.format(str(index + 1)))
            count += 1

        f.write('\n[  membrane  ]\n')  # membrane section!
        count = 1
        for index in membrane:
            if count % 10 != 0:
                f.write('{:<8s}'.format(str(index + 1)))
            else:
                f.write('{:<8s}\n'.format(str(index + 1)))
            count += 1


def write_gro_pos(pos, out, name='NA', box=None, ids=None, res=None, vel=None, ucell=None):
    """ write a .gro file from positions

    :param pos: xyz coordinates (natoms, 3)
    :param out: name of output .gro file
    :param name: name to give atoms being put in the .gro
    :param box: unitcell vectors. Length 9 list or length 3 list if box is cubic
    :param ids: name of each atom ordered by index (i.e. id 1 should correspond to atom 1)
    :param: res: name of residue for each atom
    :param: vel: velocity of each atom (natoms x 3 numpy array)
    :param: ucell: unit cell dimensions in mdtraj format (a 3x3 matrix)

    :type pos: np.ndarray
    :type out: str
    :type name: str
    :type box: list
    :type ids: list
    :type res: list
    :type vel: np.ndarray
    :type ucell: np.ndarray

    :return: A .gro file
    """

    if ucell is not None:
        box = [ucell[0, 0], ucell[1, 1], ucell[2, 2], ucell[0, 1], ucell[2, 0], ucell[1, 0], ucell[0, 2], ucell[1, 2],
               ucell[2, 0]]

    if box is None:  # to avoid mutable default
        box = [0., 0., 0.]

    with open(out, 'w') as f:

        f.write('This is a .gro file\n')
        f.write('%s\n' % pos.shape[0])

        for i in range(pos.shape[0]):
            if vel is not None:
                if ids is not None:
                    f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}{:8.4f}{:8.4f}{:8.4f}\n'.format((i + 1) % 100000, '%s' % name, '%s' % name,
                                                                            (i + 1) % 100000, pos[i, 0], pos[i, 1], pos[i, 2], vel[i, 0], vel[i, 1], vel[i, 2]))
                else:
                    f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}{:8.4f}{:8.4f}{:8.4f}\n'.format((i + 1) % 100000, '%s' % res[i], '%s' % ids[i],
                                                                            (i + 1) % 100000, pos[i, 0], pos[i, 1], pos[i, 2], vel[i, 0], vel[i, 1], vel[i, 2]))

            else:
                if ids is None:
                    f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}\n'.format((i + 1) % 100000, '%s' % name, '%s' % name,
                                                                            (i + 1) % 100000, pos[i, 0], pos[i, 1], pos[i, 2]))
                else:
                    f.write('{:5d}{:5s}{:>5s}{:5d}{:8.3f}{:8.3f}{:8.3f}\n'.format((i + 1) % 100000, '%s' % res[i], '%s' % ids[i],
                                                                            (i + 1) % 100000, pos[i, 0], pos[i, 1], pos[i, 2]))
        for i in range(len(box)):
            f.write('{:10.5f}'.format(box[i]))

        f.write('\n')
        # f.write('{:10f}{:10f}{:10f}\n'.format(0, 0, 0))


def write_em_mdp(steps, freeze=False, freeze_group='', freeze_dim='xyz', xlink=False):
    """
    Write energy minimization .mdp file
    :param steps: number of steps to take using steepest descents algorithm
    :return: Directly writes an energy minimization .mdp file
    """

    with open('em.mdp', 'w') as f:

        f.write("title = Energy Minimization\n")
        f.write("integrator = steep\n")
        f.write("nsteps = %s\n" % steps)
        f.write("cutoff-scheme = verlet\n")
        f.write("nstlist = 40\n")

        if freeze:
            f.write('freezegrps = %s\n' % freeze_group)
            dim = []
            if 'x' in freeze_dim:
                dim.append('Y')
            else:
                dim.append('N')
            if 'y' in freeze_dim:
                dim.append('Y')
            else:
                dim.append('N')
            if 'z' in freeze_dim:
                dim.append('Y')
            else:
                dim.append('N')
            f.write('freezedim = %s %s %s\n' %(dim[0], dim[1], dim[2]))

        if xlink:
            f.write('periodic-molecules = yes\n')


def save_object(obj, filename):

    with open(filename, 'wb') as output:  # Overwrites any existing file.

        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):

    with open(filename, 'rb') as f:

        return pickle.load(f)
