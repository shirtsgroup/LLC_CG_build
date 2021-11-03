#!/usr/bin/env python

import argparse
import mdtraj as md
import os
import topology


def initialize():

    parser = argparse.ArgumentParser(description='Generate topology file from coordinate file.')

    parser.add_argument('-g', '--gro', help='Name of coordinate file to write topology file for')
    parser.add_argument('-o', '--output', default='topol.top', help='Name of topology to output')
    parser.add_argument('-d', '--description', default='Simulation box', help='Description of system to put under '
                                                                              '[ system ] directive')
    parser.add_argument('-ff', '--forcefield', default='mod_gly_ion', help='Name of forcefield to use')
    parser.add_argument('-xlink', action="store_true", help='Create topology for cross-linked system')
    parser.add_argument('-xlink_topname', default='assembly.itp', help='Name of cross-linked topology')

    return parser


class SystemTopology(object):

    def __init__(self, gro, ff='mod_gly_ion', restraints=False, xlink=False, xlinked_top_name='assembly.itp',
                 fix_residues=False):
        """ Read coordinate file, identify and count different residues, and figure out where inidividual residue
        topologies are located

        :param gro: coordinate file for which to create topology
        :param ff: forcefield to use
        :param restraints: True if there are any position restraints
        :param xlink: True if system is cross-linked (Topology is specialized in this case)
        :param xlinked_top_name: Name of topology describing cross-linked system (include full path if not located in \
        same directory where this script is run)

        :type gro: str
        :type ff: str
        :type restraints: bool
        :type xlink: bool
        :type xlinked_top_name: str
        """

        if fix_residues:
            topology.fix_resnumbers(gro)

        t = md.load(gro)  # load coordinate file

        self.script_location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.top_location = "%s/../top/topologies" % self.script_location  # topology location
        self.ff_location = "%s/../top/Forcefields" % self.script_location  # forcefield location
        self.forcefield = ff  # which forcefield to use
        self.name = None
        self.atoms = [a.name for a in t.topology.atoms]  # all atom names
        self.xlink = xlink
        self.xlinked_top_name = xlinked_top_name

        if self.xlink:
            with open(xlinked_top_name, 'r') as f:
                a = []
                for line in f:
                    if line.count('[ atoms ]') == 1:
                        break
                    else:
                        a.append(line)

            molecule = 0
            while a[molecule].count('[ moleculetype ]') == 0:
                molecule += 1
            molecule += 1
            while a[molecule][0] == ';':
                molecule += 1

            self.xlink_residue = a[molecule].split()[0]
        else:
            self.xlink_residue = None

        if restraints:
            if restraints is list:
                self.restraints = [a for a in restraints]
            else:
                self.restraints = restraints
        else:
            self.restraints = False

        with open('%s/ions.txt' % self.top_location) as f:
            self.ions = []
            for line in f:
                if line[0] != '#':
                    self.ions.append(str.strip(line))

        residues = [a.residue.name for a in t.topology.atoms]

        for i, r in enumerate(residues):
            if r == 'HOH':
                residues[i] = 'SOL'

        # unique_residues = list(set(residues)) # this does not preserve order which is necessary for topology writing
        unique_residues = []
        for x in residues:
            if x not in unique_residues:
                unique_residues.append(x)

        residue_count = {}
        for i in unique_residues:
            if i in self.ions:
                natoms = 1
            else:
                natoms = topology.Residue(i).natoms
                #natoms = md.load('%s/%s.gro' % (self.top_location, i)).xyz.shape[1]  # number of atoms that make up a single residue of type i
            residue_count[i] = residues.count(i) // natoms

        self.residues = unique_residues
        self.residue_count = residue_count

    def write_top(self, name='topol.top', description='Simulation Box', restrained_top_name='restrained.itp'):
        """ Write out the topology in appropriate GROMACS format

        :param name: name of output topology file
        :param description: Description to add to [ system ] section of topology
        :param restrained_top_name: Name of topology file that includes position restraints (TODO: add this to __init__)

        :type name: str
        :type description: str
        :type restrained_top_name: str
        """

        self.name = name

        top = []
        top.append(';Forcefield\n')
        top.append('#include "%s/%s.itp"\n' % (self.ff_location,  self.forcefield))
        top.append('\n')

        restraints_included = False
        print("xlink residue", self.xlink_residue)
        ion_count = 0
        for r in self.residues:
            if r=='SOL' : r='HOH'  #the itp file should be 'HOH.itp' for water
            if r in self.ions:
                if ion_count == 0:
                    top.append(';Ion Topology\n')
                    top.append('#include "%s/ions.itp"\n' % self.top_location)
                    top.append('\n')
                ion_count += 1
            else:
                top.append(';%s Topology\n' % r)
                if self.restraints and not self.xlink: #I added the self.xlink part
                    if r in self.restraints:
                        if not restraints_included:
                            print("including ", restrained_top_name)
                            top.append('#include "%s"\n' % restrained_top_name)  # need to modify so this is not hardcoded.
                            restraints_included = True  # prevent restrained topology from being included twice
                    else:
                        top.append('#include "%s/%s.itp"\n' % (self.top_location, r))
                elif self.xlink and r == self.xlink_residue:
                    print("including ", self.xlinked_top_name)
                    top.append('#include "%s"\n' % self.xlinked_top_name)  # need to modify so this is not hardcoded.
                else:
                    top.append('#include "%s/%s.itp"\n' % (self.top_location, r))
                top.append('\n')

        top.append('[ system ]\n')
        top.append('%s\n' % description)
        top.append('\n')

        top.append('[ molecules ]\n')
        top.append(';Compounds     nmols\n')

        restraints_included = False
        for r in self.residues:
            if self.restraints and not self.xlink:
                if r in self.restraints:
                    if not restraints_included:
                        top.append('{:10s}{:>10d}\n'.format('restrained', 1))
                        restraints_included = True
                else:
                    top.append('{:10s}{:>10d}\n'.format(r, self.residue_count[r]))
            elif self.xlink and r == self.xlink_residue:
                top.append('{:10s}{:>10d}\n'.format(r, 1))
            else:
                top.append('{:10s}{:>10d}\n'.format(r, self.residue_count[r]))

        with open(self.name, 'w') as f:
            for line in top:
                f.write(line)

    def add_residue(self, residue, n=1, write=False, topname='topol.top', top_description='Simulation Box'):
        """ Add molecule(s) of a single residue to the topology

        :param residue: name of residue object
        :param n: number of molecules to add
        :param write: write new topology file
        :param topname: name of output topology if written
        :param top_description: system description to be written into top if desired

        :type residue: LLC_Membranes.llclib.topology.Residue object
        :type n: int
        :type write: bool
        :type topname: str
        :type top_description: str
        """

        resname = residue.name

        if resname == 'HOH' : resname='SOL' #SOL is the resname in HOH.itp

        if residue.is_ion:
            self.ions.append(resname)

        if resname not in self.residues:
            self.residues.append(resname)
            self.residue_count[resname] = n
        else:
            self.residue_count[resname] += n

        if write:
            self.write_top(name=topname, description=top_description)

    def remove_residue(self, residue, n=1, write=False, topname='topol.top', top_description='Simulation Box'):
        """ Remove a residue from the topology

        :param residue: name of residue object
        :param n: number of molecules to add
        :param write: write new topology file
        :param topname: name of output topology if written
        :param top_description: system description to be written into top if desired

        :type residue: LLC_Membranes.llclib.topology.Residue object
        :type n: int
        :type write: bool
        :type topname: str
        :type top_description: str
        """

        resname = residue.resname
        self.residue_count[resname] -= n

        if write:
            self.write_top(name=topname, description=top_description)


if __name__ == "__main__":

    args = initialize().parse_args()
    t = SystemTopology(args.gro, ff=args.forcefield, xlink=True)
    t.write_top(name=args.output, description=args.description)
