import parmed as pmd
#gmx_top = pmd.load_file("topol.top", xyz='solvated_final.gro', parametrize=False)
#gmx_top.save('solvated_final.psf')


gmx_top = pmd.load_file("topol1.top", xyz='initial.gro', parametrize=False)
gmx_top.save('initial.psf')
