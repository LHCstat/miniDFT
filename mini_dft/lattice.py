import constants as const
import numpy as np
#A应该竖着放向量还是横着可能需要考虑一下
#坐标转化函数也需要考虑
class Lattice:
    #A是晶格向量矩阵,输入POSCAR里的上面
    def __init__(self, A):
        self.A = A   # 假设已为 Bohr
        self.volume = abs(np.linalg.det(A))
        #self.volume_ang3 = self.volume * (const.BOHR_TO_ANGSTROM ** 3)
        self.B = np.linalg.inv(A).T*2*np.pi

# a = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
# b = Lattice(a)
#print(b.B)
def fractional_to_cartesian(fractional_coords, lattice):
    """
    将分数坐标转换为笛卡尔坐标。
    
    参数:
    fractional_coords: 分数坐标 (numpy 数组)
    lattice: Lattice 对象
    
    返回:
    笛卡尔坐标 (numpy 数组)
    """
    return np.dot( lattice.A,fractional_coords)
def cartesian_to_fractional(cartesian_coords, lattice):
    """
    将笛卡尔坐标转换为分数坐标。
    
    参数:
    cartesian_coords: 笛卡尔坐标 (numpy 数组)
    lattice: Lattice 对象
    
    返回:
    分数坐标 (numpy 数组)
    """
    return np.dot(lattice.B.T/(2*np.pi),cartesian_coords)
# print(b.B)
# print(np.dot(b.A.T,b.B))
# print(np.dot(a,np.linalg.inv(a).T*2*np.pi))

