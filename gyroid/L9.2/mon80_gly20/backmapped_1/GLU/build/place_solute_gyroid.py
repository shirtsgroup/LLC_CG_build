#!/usr/bin/env python

import argparse
import mdtraj as md
import numpy as np
from LLC_Membranes.llclib import file_rw, transform, atom_props, topology
from LLC_Membranes.setup.gentop import SystemTopology
import subprocess
import os
import tqdm
import matplotlib.path as mplpath
from scipy import spatial

script_location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))



def initialize():
    parser = argparse.ArgumentParser(description='Convenience script for adding solutes to pores')

    parser.add_argument('-g', '--gro', default='npt_long.gro', type=str, help='Coordinate file to which solutes will be'
                                                                              'added')
    parser.add_argument('-o', '--out', default='solvated_ions_20.gro', type=str, help='Name of output topology file')
    parser.add_argument('-r', '--solute_resname', default='GLU', type=str, help='Name of solute residue that is being'
                                                                                'added')
    parser.add_argument('-n', '--nsolutes', default=20, type=int, help='Number of solute molecules to add per pore')
    parser.add_argument('-mdps', '--generate_mdps', action="store_true", help='Create input .mdp files')
    parser.add_argument('-noxlink', action="store_false", help='If the system is not cross-linked, add this flag')
    parser.add_argument('-mpi', '--mpi', default=False, help="Specify number of MPI processes")

    return parser


def concentration_to_nsolute(conc, box_vectors, solute):
    """
    :param conc: (float) desired solute concentration (M)
    :param box_vectors: (numpy array, (3, 3)) box vectors. Each row represents a box vector.
    :param solute: mdtraj trajectory object generated from solute configuration file (.gro)
    :return: (int) number of solute molecules to add to box to achieve desired concentration
    """

    V = np.dot(box_vectors[2, :], np.cross(box_vectors[0, :], box_vectors[1, :]))  # box volume (nm^3)
    V *= 1 * 10 ** -24  # convert to L
    mols_solute = conc * V  # number of mols of solvent to add

    # mw = 0  # molecular weight (grams)
    # for a in solute.topology.atoms:
    #     mw += atom_props.mass[a.name]

    mass_to_add = solute.mw * mols_solute

    NA = 6.022 * 10 ** 23  # avogadro's number
    mass_solute = solute.mw / NA  # mass of a single solutes (grams)

    nsolute = int(mass_to_add / mass_solute)  # number of solute molecules to add

    actual_concentration = nsolute / (NA * V)  # mol/L

    return nsolute, actual_concentration


def net_charge(nsolute, solutes):
    """
    :param nsolute: list of number of solutes to be added
    :param solutes: list of solute objects
    :return: net charge of system after addition of nsolute
    """

    net_charge = 0
    for i, n in enumerate(nsolute):
        net_charge += n * solutes[i].charge

    return net_charge


def put_in_box(pt, x_box, y_box, m, angle):
    """
    :param pt: The point to place back in the box
    :param x_box: length of box in x dimension
    :param y_box: length of box in y dimension
    :param m: slope of box vector
    :param angle: angle between x axis and y box vector
    :return: coordinate shifted into box
    """

    b = - m * x_box  # y intercept of box vector that does not pass through origin (right side of box)
    if pt[1] < 0:
        pt[:2] += [np.cos(angle) * x_box, np.sin(angle) * x_box]  # if the point is under the box
    if pt[1] > y_box:
        pt[:2] -= [np.cos(angle) * x_box, np.sin(angle) * x_box]
    if pt[0] < 0:  # if the point is on the left side of the box
        pt[0] += x_box
    if pt[0] > x_box:  # if the point is on the right side of the box
        pt[0] -= x_box
    # if pt[1] > m*pt[0]:  # if the point is on the left side of the box
    # pt[0] += x_box
    # if pt[1] < (m*pt[0] + b):  # if the point is on the right side of the box
    # pt[0] -= x_box
    return pt


def trace_pores(pos, box, npoints, npores=1, progress=True, save=True, savename='spline.pl'):
    """
    Find the line which traces through the center of the pores
    :param pos: positions of atoms used to define pore location (args.ref) [natoms, 3]
    :param box: xy box vectors, [2, 2], mdtraj format (t.unitcell_vectors)
    :param npoints: number of points for spline in each pore
    :param npores: number of pores in unit cell (assumed that atoms are number sequentially by pore. i.e. pore 1 atom
    numbers all precede those in pore 2)
    :param progress: set to True if you want a progress bar to be shown
    :param save: save spline as pickled object
    :param savename: path to spline. If absolute path is not provided, will look in current directory

    :type pos: np.ndarray
    :type box: np.ndarray
    :type npoints: int
    :type npores: int
    :type progress: bool
    :type save: bool
    :type savename: str

    :return: points which trace the pore center
    """
    try:
        print('Attempting to load spline ... ', end='', flush=True)
        spline = file_rw.load_object(savename)
        print('Success!')

        return spline[0], spline[1]

    except FileNotFoundError:

        print('%s not found ... Calculating spline' % savename)

        single_frame = False
        if np.shape(pos.shape)[0] == 2:
            pos = pos[np.newaxis, ...]  # add a new axis if we are looking at a single frame
            box = box[np.newaxis, ...]
            single_frame = True

        nframes = pos.shape[0]
        atoms_p_pore = int(pos.shape[1] / npores)  # atoms in each pore

        v = np.zeros([nframes, 4, 2])  # vertices of unitcell box
        bounds = []
        v[:, 0, :] = [0, 0]
        v[:, 1, 0] = box[:, 0, 0]
        v[:, 3, :] = np.vstack((box[:, 1, 0], box[:, 1, 1])).T
        v[:, 2, :] = v[:, 3, :] + np.vstack((box[:, 0, 0], np.zeros([nframes]))).T
        center = np.vstack((np.mean(v[..., 0], axis=1), np.mean(v[..., 1], axis=1), np.zeros(nframes))).T
        for t in range(nframes):
            bounds.append(mplpath.Path(v[t, ...]))  # create a path tracing the vertices, v

        angle = np.arcsin(
            box[:, 1, 1] / box[:, 0, 0])  # specific to case where magnitude of x and y box lengths are equal
        # Subin: what if box[:, 1, 1]>box[:, 0, 0]
        angle = np.where(box[:, 1, 0] < 0, angle + np.pi / 2, angle)  # haven't tested this well yet

        m = (v[:, 2, 1] - v[:, 0, 1]) / (
                    v[:, 2, 0] - v[:, 0, 0])  # slope from points connecting first and third vertices

        centers = np.zeros([nframes, npores, npoints, 3])
        bin_centers = np.zeros([nframes, npores, npoints])
        for t in tqdm.tqdm(range(nframes), disable=(not progress)):
            for p in range(npores):

                pore = pos[t, p * atoms_p_pore:(p + 1) * atoms_p_pore,
                       :]  # coordinates for atoms belonging to a single pore

                while np.min(pore[:, 2]) < 0 or np.max(pore[:, 2]) > box[t, 2, 2]:
                    # because cross-linked configurations can extend very far up and down
                    pore[:, 2] = np.where(pore[:, 2] < 0, pore[:, 2] + box[t, 2, 2], pore[:, 2])
                    pore[:, 2] = np.where(pore[:, 2] > box[t, 2, 2], pore[:, 2] - box[t, 2, 2], pore[:, 2])

                _, bins = np.histogram(pore[:, 2], bins=npoints)  # bin z-positions

                section_indices = np.digitize(pore[:, 2], bins)  # list that tells which bin each atom belongs to
                bin_centers[t, p, :] = [(bins[i] + bins[i + 1]) / 2 for i in range(npoints)]

                for l in range(1, npoints + 1):

                    atom_indices = np.where(section_indices == l)[0]

                    before = pore[atom_indices[0], :]  # choose the first atom as a reference

                    shift = transform.translate(pore[atom_indices, :], before, center[t, :])
                    # shift everything to towards the center

                    for i in range(shift.shape[0]):  # check if the points are within the bounds of the unitcell

                        while not bounds[t].contains_point(shift[i, :2]):
                            shift[i, :] = put_in_box(shift[i, :], box[t, 0, 0], box[t, 1, 1], m[t], angle[t])
                            # if its not in the unitcell, shift it so it is

                    c = [np.mean(shift, axis=0)]

                    centers[t, p, l - 1, :] = transform.translate(c, center[t, :],
                                                                  before)  # move everything back to where it was

                    while not bounds[t].contains_point(
                            centers[t, p, l - 1, :]):  # make sure everything is in the box again
                        centers[t, p, l - 1, :] = put_in_box(centers[t, p, l - 1, :], box[t, 0, 0], box[t, 1, 1], m[t],
                                                             angle[t])

        if single_frame:
            return centers[0, ...]  # doesn't return bin center yet

        else:

            if save:
                file_rw.save_object((centers, bin_centers), savename)

            return centers, bin_centers


def placement(z, pts, box):
    """
    :param z: z location where solute should be placed
    :param pts: points which run through the pore
    :return: location to place solute
    """
    # check if point is already in the spline
    if z in pts[:, 2]:
        ndx = np.where(pts[:, 2] == z)[0][0]
        return pts[ndx, :]

    # otherwise interpolate between closest spline points
    else:
        v = np.zeros([4, 2])  # vertices of unitcell box
        v[0, :] = [0, 0]
        v[1, :] = [box[0, 0], 0]
        v[3, :] = [box[1, 0], box[1, 1]]
        v[2, :] = v[3, :] + [box[0, 0], 0]
        center = [np.mean(v[:, 0]), np.mean(v[:, 1]), 0]  # geometric center of box

        bounds = mplpath.Path(v)  # create a path tracing the vertices, v

        angle = np.arccos(box[1, 1] / box[0, 0])  # angle of monoclinic box
        if box[1, 0] < 0:  # the case of an obtuse angle
            angle += np.pi / 2

        m = (v[3, 1] - v[0, 1]) / (
                    v[3, 0] - v[0, 0])  # slope from points connecting first and fourth(SS:Third) vertices

        # shift = transform.translate(z, before, center)
        #
        # put_in_box(pt, box[0, 0], box[1, 1], m, angle)

        # find z positions, in between which solute will be placed
        lower = 0
        while pts[lower, 2] < z:
            lower += 1

        upper = pts.shape[0] - 1
        while pts[upper, 2] > z:
            upper -= 1

        limits = np.zeros([2, 3])
        limits[0, :] = pts[lower, :]
        limits[1, :] = pts[upper, :]

        shift = transform.translate(limits, limits[0, :], center)  # shift limits to geometric center of unit cell
        shift[:, 2] = [limits[0, 2], limits[1, 2]]  # keep z positions the same

        for i in range(shift.shape[0]):  # check if the points are within the bounds of the unitcell
            if not bounds.contains_point(shift[i, :2]):
                shift[i, :] = put_in_box(shift[i, :], box[0, 0], box[1, 1], m, angle)

        # Use parametric representation of line between upper and lower points to find the xy value where z is satsified
        v = shift[1, :] - shift[0, :]  # direction vector

        t = (z - shift[0, 2]) / v[2]  # solve for t since we know z
        x = shift[0, 0] + t * v[0]
        y = shift[0, 1] + t * v[1]

        place = np.zeros([1, 3])
        place[0, :] = [x, y, 0]
        place = transform.translate(place, center, limits[0, :])  # put xy coordinate back
        place[0, 2] = z

        if not bounds.contains_point(place[0, :]):  # make sure everything is in the box again
            place[0, :] = put_in_box(place[0, :], box[0, 0], box[1, 1], m, angle)

        return place[0, :]


class Solvent(object):

    def __init__(self, gro, intermediate_fname='solvate.gro', em_steps=100, p_coupling='isotropic', xlink=False,
                 xlinked_topname='assembly.itp'):
        """
        :param gro: configuration of solvent
        :param intermediate_fname : name of intermediate .gro files if placing solute in box
        :param em_steps : number of energy minimization steps if placing solute in box
        """

        self.t = md.load(gro)
        self.box_vectors = self.t.unitcell_vectors[0, :, :]  # box vectors

        self.xlink = xlink

        # parallelization
        self.mpi = False  # use mpi / gpu acceleration
        self.np = 1  # number of parallel process

        self.box_gromacs = [self.box_vectors[0, 0], self.box_vectors[1, 1], self.box_vectors[2, 2],
                            self.box_vectors[0, 1], self.box_vectors[2, 0], self.box_vectors[1, 0],
                            self.box_vectors[0, 2], self.box_vectors[1, 2],
                            self.box_vectors[2, 0]]  # box in gromacs format

        self.positions = self.t.xyz[0, :, :]  # positions of all atoms
        self.residues = []
        self.names = []
        self.top = SystemTopology(gro, xlink=self.xlink, xlinked_top_name=xlinked_topname)
        self.intermediate_fname = intermediate_fname
        self.em_steps = em_steps

        # data specifically required for adding solutes to pores
        self.pore_spline = None
        self.water = [a.index for a in self.t.topology.atoms if a.residue.name == 'HOH' and a.name == 'O']
        self.water_top = topology.Solute('SOL')

        # because mdtraj changes the names
        for a in self.t.topology.atoms:
            if a.residue.name == 'HOH':
                self.residues.append('SOL')
                if a.name == 'O':
                    self.names.append('OW')
                elif a.name == 'H1':
                    self.names.append('HW1')
                elif a.name == 'H2':
                    self.names.append('HW2')
            else:
                self.residues.append(a.residue.name)
                self.names.append(a.name)

    def place_solute(self, solute, placement_point, random=False, freeze=False, rem=.5):

        """
        Place solute at desired point and energy minimze the system
        :param solute: name of solute object (str)
        :param placement_point: point to place solute (np.array([3])
        :param random: place solute at random point in box (bool)
        :param freeze: freeze all atoms outside rem during energy minimization (bool)
        :param rem: radius from placement_point within which atoms will NOT be frozen (float, nm)
        :return:
        """

        # randomly rotate the molecule and then tranlate it to the placement point
        solute_positions = transform.random_orientation(solute.xyz[0, ...], solute.xyz[0, 0, :] -
                                                        solute.xyz[0, 1, :], placement_point)
        self.positions = np.concatenate((self.positions, solute_positions))  # add to array of positions
        self.residues += solute.residues  # add solute residues to list of all residues
        # self.names += [solute.names.get(i) for i in range(1, solute.natoms + 1)]  # add solute atom names to all names
        self.names += solute.names
        solute.name = solute.resname
        self.top.add_residue(solute, write=True)  # add 1 solute to topology

        # write new .gro file
        file_rw.write_gro_pos(self.positions, self.intermediate_fname, box=self.box_gromacs, ids=self.names,
                              res=self.residues)

        if freeze:
            self.freeze_ndx(solute_placement_point=placement_point, res=solute.resname)

        nrg = self.energy_minimize(self.em_steps, freeze=freeze)
        print("energy=", nrg)
        if nrg >= 0:
            self.revert(solute)
            if random:
                self.place_solute_random(solute)
            else:
                # self.remove_water(placement_point, 3)
                self.place_solute(solute, placement_point, freeze=True)
        else:
            p3 = subprocess.Popen(["cp", "em.gro", "%s" % self.intermediate_fname])
            p3.wait()
            self.positions = md.load('%s' % self.intermediate_fname).xyz[0, :, :]  # update positions

    def place_solute_random(self, solute):
        """
        :param solute: Solute object generated from solute configuration file (.gro)
        """
        placement_point = self.random_point_box()  # where to place solute
        self.place_solute(solute, placement_point, random=True)

    def place_solute_pores(self, solute, z=None, layers=40, pores=1, ref=['C22']):

        """
        Place solute in middle of pores at given z location
        :param solute: solute object
        :param z: z location of solute center of mass (float)
        :param layers: number of layers in system (when initial configuration was set up) (int)
        :param pores: number of pores in which to place solutes (int)
        :param ref: reference atoms used to define pore center
        :return:
        """

        ref = [a.index for a in self.t.topology.atoms if a.name in ref]

        # redo each time because positions change slightly upon energy minimization
        self.pore_spline = trace_pores(self.positions[ref, :], self.t.unitcell_vectors[0, ...], layers,
                                       progress=True, npores=pores)
        print("pore atoms: ", len(ref))
        # format z so that it is an array
        if type(z) is float or type(z) is np.float64:
            z = np.array([z for i in range(pores)])

        for i in tqdm.tqdm(range(pores)):
            placement_point = placement(z[i], self.pore_spline[i, ...], self.box_vectors[:2, :2])
            print("placing solute at", placement_point)
            self.place_solute(solute, placement_point, freeze=True)

    def energy_minimize(self, steps, freeze=False, freeze_group='Freeze', freeze_dim='xyz'):
        """
        Energy minimize a configuration
        :param steps: number of steepest descent energy minimization steps to take
        :return: coordinates of energy minimized structure, updated coordinates of reference atoms
        """

        p4 = subprocess.Popen(
            ["mv", "em.tpr", "em.trr", "em.edr", "em.log", "em.mdp", "./trash"])  # remove previous files
        p4.wait()  # remove previos

        # write em.mdp with a given number of steps
        file_rw.write_em_mdp(steps, freeze=freeze, freeze_group='Freeze', freeze_dim='xyz', xlink=self.xlink)

        if freeze:
            if self.mpi:
                p1 = subprocess.Popen(
                    ["mpirun", "-np", "1", "gmx_mpi", "grompp", "-p", "topol.top", "-f", "em.mdp", "-o", "em", "-c",
                     "%s" % self.intermediate_fname, "-n", "freeze_index.ndx"], stdout=open(os.devnull, 'w'),
                    stderr=subprocess.STDOUT)  # generate atomic level input file
            else:
                p1 = subprocess.Popen(
                    ["gmx", "grompp", "-p", "topol.top", "-f", "em.mdp", "-o", "em", "-c",
                     "%s" % self.intermediate_fname,
                     "-n", "freeze_index.ndx"],
                    stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT)  # generate atomic level input file
        else:
            if self.mpi:
                p1 = subprocess.Popen(
                    ["mpirun", "-np", "1", "gmx_mpi", "grompp", "-p", "topol.top", "-f", "em.mdp", "-o", "em", "-c",
                     "%s" % self.intermediate_fname], stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT)
                p1.wait()
            else:
                p1 = subprocess.Popen(
                    ["gmx", "grompp", "-p", "topol.top", "-f", "em.mdp", "-o", "em", "-c",
                     "%s" % self.intermediate_fname],
                    stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT)  # generate atomic level input file
        p1.wait()

        if self.mpi:
            p2 = subprocess.Popen(["mpirun", "-np", "%s" % self.np, "gmx_mpi", "mdrun", "-deffnm", "em"],
                                  stdout=open(os.devnull, 'w'), stderr=subprocess.STDOUT)  # run energy minimization
        else:
            p2 = subprocess.Popen(["gmx", "mdrun", "-deffnm", "em"], stdout=open(os.devnull, 'w'),
                                  stderr=subprocess.STDOUT)  # run energy minimization
        p2.wait()
        print("Energy Minimization done!!!!!!")
        nrg = subprocess.check_output(
            ["awk", "/Potential Energy/ {print $4}", "em.log"])  # get Potential energy from em.log

        try:
            return float(nrg.decode("utf-8"))
        except ValueError:
            return 0  # If the system did not energy minimize, the above statement will not work because nrg will be an
            # empty string. Make nrg=0 so placement gets attempted again

    def freeze_ndx(self, solute_placement_point=None, rem=None, res=None):
        """
        Write an index file for atoms to be frozen
        :param solute_placement_point: xyz position of where water molecule was placed
        :param rem: spherical radius measured from water molecule placement point outside which all atoms will be frozen
        :param res: freeze this residue and no other atoms (can be combined with rem option)
        :return: index file with indices of atoms to be frozen
        """

        freeze_indices = []
        if rem:
            pts = spatial.cKDTree(self.positions).query_ball_point(solute_placement_point, rem)
            freeze_indices = [a.index for a in self.t.topology.atoms if a.index not in pts]
        elif res:
            freeze_indices += [a for a in range(len(self.residues)) if self.residues[a] == res]
        else:
            print('WARNING: No valid options supplied in order to determine freeze indices. Specify rem or res.')

        with open('freeze_index.ndx', 'w') as f:

            f.write('[ Freeze ]\n')
            for i, entry in enumerate(freeze_indices):
                if (i + 1) % 15 == 0:
                    f.write('{:5d}\n'.format(entry + 1))
                else:
                    f.write('{:5d} '.format(entry + 1))

    def random_point_box(self):
        """
        :param box_vectors: (numpy array, (3, 3)) box vectors. Each row represents a box vector.
        :return: (numpy array, (3)) coordinates of a randomly chosen point that lies in box
        """

        A = self.box_vectors[0, :]  # x box vector
        B = self.box_vectors[1, :]  # y box vector
        C = self.box_vectors[2, :]  # z box vector
        u, v, w = np.random.rand(3)  # generate 3 random numbers between 0 and 1
        pt = np.array([0, 0, 0]) + u * A + v * B + w * C  # places point inside 3D box defined by box vector A, B and C

        return pt

    def revert(self, solute):
        """
        Revert system to how it was before solute addition
        """
        n = -solute.natoms
        self.positions = self.positions[:n, :]
        self.residues = self.residues[:n]
        self.names = self.names[:n]
        self.top.add_residue(solute, n=-1, write=False)  # subtract a solute from the topology

    def write_config(self, name='out.gro'):
        """
        Write .gro coordinate file from current positions
        :param name: name of coordinate file to write (str)
        """
        # write new .gro file
        file_rw.write_gro_pos(self.positions, name, box=self.box_gromacs, ids=self.names, res=self.residues)

    def remove_water(self, point, n):

        """
        remove n water molecules closest to point
        """

        tree = spatial.cKDTree(self.positions[self.water, :])
        rm = []

        nn = tree.query(point, k=n)[1]
        for j in nn:
            rm.append(self.water[j])
            rm.append(self.water[j] + 1)
            rm.append(self.water[j] + 2)

        # update relevant arrays
        self.positions = np.delete(self.positions, rm, axis=0)
        self.residues = [self.residues[x] for x in range(len(self.residues)) if x not in rm]
        self.names = [self.names[x] for x in range(len(self.names)) if x not in rm]
        self.water = [i for i, x in enumerate(self.residues) if x == 'SOL' and self.names[i] == 'OW']

        self.top.remove_residue(self.water_top, n, write=True)


# Revamped in llclib.topology (SS: I reinstate it)
class Solute(object):

    def __init__(self, name):

        self.is_ion = False
        # check if residue is an ion
        #with open('%s/../top/topologies/ions.txt' % script_location) as f:
        with open('ions.txt') as f:
            ions = []
            for line in f:
                if line[0] != '#':
                    ions.append(str.strip(line))

        if name in ions:
            self.is_ion = True
            self.residues = [name]
            self.names = [name]
            self.xyz = np.zeros([1, 1, 3])
            self.xyz[0, 0, :] = [0, 0, 0]
            self.natoms = 1
            self.mw = atom_props.mass[name]
            self.charge = atom_props.charge[name]
            self.resname = name
        else:
            try:
                t = md.load('%s.pdb' % name,
                            standard_names=False)  # see if there is a solute configuration in this directory
            except OSError:
                try:
                    t = md.load('%s/../top/topologies/%s.pdb' % (script_location, name), standard_names=False)
                    # located with all of the other topologies
                except OSError:
                    print('No residue %s found' % name)
                    exit()

            try:
                f = open('%s.itp' % name, 'r')
            except FileNotFoundError:
                try:
                    f = open('%s/../top/topologies/%s.itp' % (script_location, name), 'r')
                except FileNotFoundError:
                    print('No topology %s.itp found' % name)

            itp = []
            for line in f:
                itp.append(line)

            f.close()

            self.natoms = t.n_atoms

            atoms_index = 0
            while itp[atoms_index].count('[ atoms ]') == 0:
                atoms_index += 1

            atoms_index += 2
            self.charge = 0
            for i in range(self.natoms):
                self.charge += float(itp[atoms_index + i].split()[6])

            self.residues = [a.residue.name for a in t.topology.atoms]
            self.resname = self.residues[0]
            self.names = [a.name for a in t.topology.atoms]
            self.xyz = t.xyz

            self.mw = 0  # molecular weight (grams)
            for a in t.topology.atoms:
                self.mw += atom_props.mass[a.name]

            self.com = np.zeros([3])  # center of mass of solute
            for i in range(self.xyz.shape[1]):
                self.com += self.xyz[0, i, :] * atom_props.mass[self.names[i]]
            self.com /= self.mw


if __name__ == "__main__":

    os.environ["GMX_MAXBACKUP"] = "-1"  # stop GROMACS from making backups

    args = initialize().parse_args()

    solvent = Solvent('%s' % args.gro, xlink=args.noxlink)
    solute = Solute('%s' % args.solute_resname)

    if args.mpi:
        solvent.mpi = True
        solvent.np = int(args.mpi)

    zbox = solvent.box_vectors[2, 2]
    z = np.linspace(0, zbox, args.nsolutes * 2 + 1)[1::2]  # equally space residues
    for i in range(args.nsolutes):
        solvent.place_solute_pores(solute, z=z[i])

    solvent.write_config(name='%s' % args.out)

    if args.generate_mdps:
        mdp = genmdp.SimulationMdp('%s' % args.out, length=5000, barostat='berendsen', xlink=args.noxlink, frames=50)
        mdp.write_em_mdp()
        mdp.write_npt_mdp(out='berendsen')

        mdp = genmdp.SimulationMdp('%s' % args.out, length=1000000, barostat='Parrinello-Rahman', xlink=args.noxlink,
                                   genvel='no', frames=2000)
        mdp.write_npt_mdp(out='PR')

    # put everything in monoclinic cell
    pipe = subprocess.Popen(['echo', '0'], stdout=subprocess.PIPE)
    put_in_box = "gmx trjconv -f %s -o %s -pbc atom -ur tric -s em.tpr" % (args.out, args.out)
    subprocess.Popen(put_in_box.split(), stdin=pipe.stdout)
