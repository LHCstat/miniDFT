import ionic_potential
import hartree
import xc
def get_potential(n,system):
    #输入电子密度n和系统的以获取离子势能以获得总势能
    potential = ionic_potential.ionic_potential(system) + hartree.hartree(n) + xc.xc(n)
    return potential
