import fft_grid
def kinetic_energy(c,k):
    #看下这个动能项有没有搞错
    kinetic = c*0.5*(k+c)**2
    return kinetic

def applyHc(potential,c,k):
    #输入势能和c，返回作用结果，在k空间
    #需要研究反向fft得到的x轴存储范围
    kinetic = kinetic_energy(c,k)
    psi_r = fft_grid.reciprocal_to_real(c)
    veff = fft_grid.reciprocal_to_real(potential)
    V_r = veff * psi_r
    V_G = fft_grid.real_to_reciprocal(V_r)

    return kinetic + V_G
