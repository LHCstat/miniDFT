import i_o
#定义了系统类，每次需要调用数据时都直接用这个类就行了
class system:
    def __init__(self, filename):
        self.lattice = i_o.get_lattice(filename)
        self.atoms = i_o.get_atoms(filename)
        self.electron_number = i_o.get_electron_number(filename)
        self.ENCUT = i_o.get_ENCUT(filename)
        self.kpoints = i_o.get_kpoints(filename)
        self.potential = i_o.get_potential(filename)
        self.SCF_params = i_o.get_SCF_params(filename)
        self.DOS_params = i_o.get_DOS_params(filename)
        self.initial_density = i_o.get_initial_density(filename)
        self.bands = i_o.get_bands(filename)
# filename = 'input.yaml'
# system = system(filename)
# lattice = system.lattice
# atoms = system.atoms
# electron_number = system.electron_number
# ENCUT = system.ENCUT
# kpoints = system.kpoints
# potential = system.potential
# SCF_params = system.SCF_params
# DOS_params = system.DOS_params
# print("lattice:", lattice)
# print("atoms:", atoms)
# print("electron_number:", electron_number)
# print("ENCUT:", ENCUT)
# print("kpoints:", kpoints)
# print("potential:", potential)
# print("SCF_params:", SCF_params)
# print("DOS_params:", DOS_params)
