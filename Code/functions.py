import numpy as np
from .core import ConstrainedMultiObjective
from nevergrad.parametrization import parameter as p


# Test Functions Reference Website: https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization

"""
=========================
2 Objective Functions with 1 variables
=========================
"""
# Schaffer Function N. 1
class Schaffer1(ConstrainedMultiObjective):
    def __init__(self, bound=1e1):
        self.bound = bound
        param = p.Array(shape=(1,), lower=np.array([-self.bound]), upper=np.array([self.bound]))
        super().__init__("Schaffer Function N. 1", param)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]**2
        f2 = (x[0] - 2)**2
        return np.array([f1, f2])

# Schaffer Function N. 2
class Schaffer2(ConstrainedMultiObjective):
    def __init__(self):
        param = p.Array(shape=(1,), lower=np.array([-5]), upper=np.array([10]))
        super().__init__("Schaffer Function N. 2", param)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x[0] <= 1:
            f1 = x[0]**2
        elif 1 < x[0] <= 3:
            f1 = x[0] - 2
        elif 3 < x[0] <= 4:
            f1 = 4 - x[0]
        elif 4 < x[0]:
            f1 = x[0] - 4
        f2 = (x[0] - 5)**2
        return np.array([f1, f2])

"""
=========================
2 Objective Functions with 2 variables
=========================
"""
# BinhKorn
class BinhKorn(ConstrainedMultiObjective):
    def __init__(self):
        param = p.Array(shape=(2,), lower=np.array([0, 0]), upper=np.array([5, 3]))
        super().__init__("Binh and Korn function", param)
        param.register_cheap_constraint(self.constraint1)
        param.register_cheap_constraint(self.constraint2)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        f1 = 4 * x[0]**2 + 4 * x[1]**2
        f2 = (x[0] - 5)**2 + (x[1] - 5)**2
        return np.array([f1, f2])

    def constraint1(self, x: np.ndarray) -> bool:
        return (x[0]-5)**2 + x[1]**2 >= 25

    def constraint2(self, x: np.ndarray) -> bool:
        return (x[0]-8)**2 + (x[1]+3)**2 >= 7.7

# Chankong and Haimes
class ChankongHaimes(ConstrainedMultiObjective):
    def __init__(self):
        param = p.Array(shape=(2,), lower=np.array([-20, -20]), upper=np.array([20, 20]))
        super().__init__("Chankong and Haimes function", param)
        param.register_cheap_constraint(self.constraint1)
        param.register_cheap_constraint(self.constraint2)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        f1 = 2 + (x[0] - 2)**2 + (x[1] - 1)**2
        f2 = 9 * x[0] - (x[1] - 1)**2
        return np.array([f1, f2])

    def constraint1(self, x: np.ndarray) -> bool:
        return x[0]**2 + x[1]**2 <= 225

    def constraint2(self, x: np.ndarray) -> bool:
        return x[0] - 3*x[1] + 10 <= 0

# Poloni's Two Objective Function
class PoloniTwoObjective(ConstrainedMultiObjective):
    def __init__(self):
        param = p.Array(shape=(2,), lower=np.array([-np.pi, -np.pi]), upper=np.array([np.pi, np.pi]))
        super().__init__("Poloni's Two Objective function", param)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        A1 = 0.5*np.sin(1) - 2*np.cos(1) + np.sin(2) - 1.5*np.cos(2)
        A2 = 1.5*np.sin(1) - np.cos(1) + 2*np.sin(2) - 0.5*np.cos(2)
        B1 = 0.5*np.sin(x[0]) - 2*np.cos(x[0]) + np.sin(x[1]) - 1.5*np.cos(x[1])
        B2 = 1.5*np.sin(x[0]) - np.cos(x[0]) + 2*np.sin(x[1]) - 0.5*np.cos(x[1])
        f1 = 1 + (A1-B1)**2 + (A2-B2)**2
        f2 = (x[0]+3)**2 + (x[1]+1)**2
        return np.array([f1, f2])

# TestFunction4
class TestFunction4(ConstrainedMultiObjective):
    def __init__(self):
        param = p.Array(shape=(2,), lower=np.array([-7, -7]), upper=np.array([4, 4]))
        super().__init__("Test Function 4", param)
        param.register_cheap_constraint(self.constraint1)
        param.register_cheap_constraint(self.constraint2)
        param.register_cheap_constraint(self.constraint3)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]**2 - x[1]
        f2 = -0.5 * x[0] - x[1] - 1
        return np.array([f1, f2])
        
    def constraint1(self, x: np.ndarray) -> bool:
        return 6.5 - (x[0]/6) - x[1] >= 0

    def constraint2(self, x: np.ndarray) -> bool:
        return 7.5 - 0.5*x[0] - x[1] + 10 >= 0

    def constraint3(self, x: np.ndarray) -> bool:
        return 30 - 5 * x[0] - x[1] >= 0

#CTP1
class CTP1(ConstrainedMultiObjective):
    def __init__(self):
        param = p.Array(shape=(2,), lower=np.array([0, 0]), upper=np.array([1, 1]))
        super().__init__("CTP1 Function", param)
        param.register_cheap_constraint(self.constraint1)
        param.register_cheap_constraint(self.constraint2)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        f2 = (1 + x[1]) * (1 - np.sqrt(f1 / 1 + x[1]))
        return np.array([f1, f2])

    def constraint1(self, x: np.ndarray) -> bool:
        f1, f2 = self.__call__(x)
        return f2 / (0.858 * np.exp(-0.541 * f1)) >= 1

    def constraint2(self, x: np.ndarray) -> bool:
        f1, f2 = self.__call__(x)
        return f2 / (0.728 * np.exp(-0.295 * f1)) >= 1

#ConstrEx
class ConstrEx(ConstrainedMultiObjective):
    def __init__(self):
        param = p.Array(shape=(2,), lower=np.array([0.1, 0]), upper=np.array([1, 5]))
        super().__init__("Constr-Ex Problem", param)
        param.register_cheap_constraint(self.constraint1)
        param.register_cheap_constraint(self.constraint2)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        f2 = (1 + x[1]) / x[0]
        return np.array([f1, f2])
            
    def constraint1(self, x: np.ndarray) -> bool:
        return 9 * x[0] + x[1] >= 6

    def constraint2(self, x: np.ndarray) -> bool:
        return 9 * x[0] - x[1] >= 1


"""
=========================
2 Objective Functions with more than 2 variables
=========================
"""
# Fonseca–Fleming Function
class FonsecaFleming(ConstrainedMultiObjective):
    def __init__(self, dimension=3):
        self.dimension = dimension
        param = p.Array(shape=(dimension,), lower=np.full(shape=dimension, fill_value=-4), upper=np.full(shape=dimension, fill_value=4)) 
        super().__init__("Fonseca-Fleming function", param)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        sum1 = np.sum((x - 1/np.sqrt(n))**2)
        f1 = 1 - np.exp(-sum1)
        sum2 = np.sum((x + 1/np.sqrt(n))**2)
        f2 = 1 - np.exp(-sum2)
        return np.array([f1, f2])

# Kursawe
class Kursawe(ConstrainedMultiObjective):
    def __init__(self, dimension=3):
        self.dimension = dimension
        param = p.Array(shape=(dimension,), lower=np.full(shape=dimension, fill_value=-5), upper=np.full(shape=dimension, fill_value=5)) 
        super().__init__("Kursawe function", param)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        f1 = sum([-10 * np.exp(-0.2 * np.sqrt(x[i]**2 + x[i+1]**2)) for i in range(self.dimension - 1)])
        f2 = sum([abs(x[i])**0.8 + 5 * np.sin(x[i]**3) for i in range(self.dimension)])
        return np.array([f1, f2])

# Zitzler–Deb–Thiele's function
# N.1
class ZDT1(ConstrainedMultiObjective):
    def __init__(self, dimension=2):
        self.dimension = dimension
        param = p.Array(shape=(dimension,), lower=np.full(shape=dimension, fill_value=0), upper=np.full(shape=dimension, fill_value=1))
        super().__init__("ZDT1 Function", param)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h
        return np.array([f1, f2])

# N.2
class ZDT2(ConstrainedMultiObjective):
    def __init__(self, dimension=2):
        self.dimension = dimension
        param = p.Array(shape=(dimension,), lower=np.full(shape=dimension, fill_value=0), upper=np.full(shape=dimension, fill_value=1))
        super().__init__("ZDT2 Function", param)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
        h = 1 - (f1 / g)**2
        f2 = g * h
        return np.array([f1, f2])

# N.3
class ZDT3(ConstrainedMultiObjective):
    def __init__(self, dimension=2):
        self.dimension = dimension
        param = p.Array(shape=(dimension,), lower=np.full(shape=dimension, fill_value=0), upper=np.full(shape=dimension, fill_value=1))
        super().__init__("ZDT3 Function", param)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
        h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
        f2 = g * h
        return np.array([f1, f2])

# N.4
class ZDT4(ConstrainedMultiObjective):
    def __init__(self, dimension=2):
        self.dimension = dimension
        param = p.Array(shape=(dimension,), lower=np.array([0] + [-5] * (dimension - 1)), upper=np.array([1] + [5] * (dimension - 1)))
        super().__init__("ZDT4 Function", param)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        f1 = x[0]
        g = 1 + 10 * (len(x) - 1) + np.sum(x[1:]**2 - 10 * np.cos(4 * np.pi * x[1:]))
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h
        return np.array([f1, f2])

# N.6
class ZDT6(ConstrainedMultiObjective):
    def __init__(self, dimension=2):
        self.dimension = dimension
        param = p.Array(shape=(dimension,), lower=np.full(shape=dimension, fill_value=0), upper=np.full(shape=dimension, fill_value=1))
        super().__init__("ZDT6 Function", param)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        f1 = 1 - np.exp(-4 * x[0]) * np.sin(6 * np.pi * x[0])**6
        g = 1 +  9 * ((np.sum(x[1:]) / (len(x) - 1))**0.25)
        h = 1 - (f1 / g)**2
        f2 = g * h
        return np.array([f1, f2])

#Osyczka and Kundu
class OsyczkaKundu(ConstrainedMultiObjective):
    def __init__(self):
        param = p.Array(shape=(6, ), 
        lower=np.array([0, 0, 1, 0, 1, 0]), 
        upper=np.array([10, 10, 5, 6, 5, 10])) 
        super().__init__("OsyczkaKundu function", param)
        param.register_cheap_constraint(self.constraint1)
        param.register_cheap_constraint(self.constraint2)
        param.register_cheap_constraint(self.constraint3)
        param.register_cheap_constraint(self.constraint4)
        param.register_cheap_constraint(self.constraint5)
        param.register_cheap_constraint(self.constraint6)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        f1 = -25*((x[0]-2)**2) - (x[1]-2)**2 - (x[2]-1)**2 - (x[3]-4)**2 - (x[4]-1)**2
        f2 = sum([x[i]**2 for i in range(6)])
        return np.array([f1, f2])

    def constraint1(self, x: np.ndarray) -> bool:
        return x[0] + x[1] -2 >= 0

    def constraint2(self, x: np.ndarray) -> bool:
        return 6 - x[0] - x[1] >= 0

    def constraint3(self, x: np.ndarray) -> bool:
        return 2 + x[0] - x[1] >= 0

    def constraint4(self, x: np.ndarray) -> bool:
        return 2 - x[0] + 3*x[1] >= 0

    def constraint5(self, x: np.ndarray) -> bool:
        return 4 - (x[2]-3)**2 - x[3] >= 0

    def constraint6(self, x: np.ndarray) -> bool:
        return (x[4]-3)**2 + x[5] - 4 >= 0

"""
=========================
3 Objective Functions with 2 variables
=========================
"""
# Viennet Function
class Viennet(ConstrainedMultiObjective):
    def __init__(self):
        param = p.Array(shape=(2,), lower=np.array([-3, -3]), upper=np.array([3, 3]))
        super().__init__("Viennet Function", param)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        f1 = 0.5*(x[0]**2 + x[1]**2) + np.sin(x[0]**2 + x[1]**2)
        f2 = (3*x[0] - 2*x[1] + 4)**2 / 8 + (x[0] - x[1] + 1)**2 / 27 + 15
        f3 = 1 / (x[0]**2 + x[1]**2 + 1) - 1.1*np.exp(-(x[0]**2 + x[1]**2))
        return np.array([f1, f2, f3])
