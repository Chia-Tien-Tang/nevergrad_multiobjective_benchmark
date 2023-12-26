import unittest
import numpy as np
from nevergrad.functions.multiobjective import functions
from tqdm import tqdm

"""Unit tests for multi-objective optimization functions in Nevergrad."""

class MultiObjectiveFunctionTests(unittest.TestCase):
    
    
    def setUpClass(self):
        self.progress_bar = tqdm(total=self.count_test_methods(), desc="Running Tests")
        
    
    def count_test_methods(self):
        return sum(1 for method in dir(self) if callable(getattr(self, method)) and method.startswith('test_'))

    def tearDown(self):
        self.progress_bar.set_description_str(f"Running Tests - {self._testMethodName}")
        
        self.progress_bar.update(1)
        self.progress_bar.refresh()
        
        if self.progress_bar.n == self.count_test_methods():
            self.progress_bar.close()

    
    """
    =========================
    2 Objective Functions with 1 variables
    =========================
    """

    # Schaffer Function N. 1
    def test_schaffer1(self, bound=1e1):
        function = functions.Schaffer1()
        x = np.array([0])
        result = function(x)
        expected = [0, 4]
        self.assertTrue(np.allclose(result, expected), f"Expected {expected}, got {result}")

    # Schaffer Function N. 2
    def test_schaffer2(self):
        function = functions.Schaffer2()
        x = np.array([0])
        result = function(x)
        expected = [0, 25]
        self.assertTrue(np.allclose(result, expected), f"Expected {expected}, got {result}")

    """
    =========================
    2 Objective Functions with 2 variables
    =========================
    """
    # BinhKorn
    def test_Binhkorn(self):
        function = functions.BinhKorn()
        x = np.array([0, 0])
        result = function(x)
        expected = [0, 50]
        self.assertTrue(np.allclose(result, expected), f"Expected {expected}, got {result}")

    # Chankong and Haimes
    def test_ChankongHaimes(self):
        function = functions.ChankongHaimes()
        x = np.array([0, 10])
        result = function(x)
        expected = [87, -81]
        self.assertTrue(np.allclose(result, expected), f"Expected {expected}, got {result}")
    
    # PoloniTwoObjectiv
    def test_PoloniTwoObjective(self):
        function = functions.PoloniTwoObjective()
        x = np.array([0, 0])
        result = function(x)
        expected = [38.17916955, 10]
        self.assertTrue(np.allclose(result, expected), f"Expected {expected}, got {result}")
    
    # TestFunction4
    def test_TestFunction4(self):
        function = functions.TestFunction4()
        x = np.array([0, 0])
        result = function(x)
        expected = [0, -1]
        self.assertTrue(np.allclose(result, expected), f"Expected {expected}, got {result}")

    #CTP1
    def test_CTP1(self):
        function = functions.CTP1()
        x = np.array([0, 0])
        result = function(x)
        expected = [0, 1]
        self.assertTrue(np.allclose(result, expected), f"Expected {expected}, got {result}")

    #ConstrEx
    def test_ConstrEx(self):
        function = functions.ConstrEx()
        x = np.array([1, 5])
        result = function(x)
        expected = [1, 6]
        self.assertTrue(np.allclose(result, expected), f"Expected {expected}, got {result}")

    """
    =========================
    2 Objective Functions with more than 2 variables
    =========================
    """
    # Fonseca–Fleming Function
    def test_FonsecaFleming(self):
        function = functions.FonsecaFleming(dimension=4)
        x = np.array([1/2, 1/2, 1/2, 1/2])
        result = function(x)
        expected = [0, 0.98168436]
        self.assertTrue(np.allclose(result, expected), f"Expected {expected}, got {result}")

    # Kursawe
    def test_Kursawe(self):
        function = functions.Kursawe(dimension=3)
        x = np.array([0, 0, 0])
        result = function(x)
        expected = [-20, 0]
        self.assertTrue(np.allclose(result, expected), f"Expected {expected}, got {result}")

    # Zitzler–Deb–Thiele's function
    # N.1
    def test_ZDT1(self):
        function = functions.ZDT1(dimension=5)
        x = np.array([0, 0, 0, 0, 0])
        result = function(x)
        expected = [0, 1]
        self.assertTrue(np.allclose(result, expected), f"Expected {expected}, got {result}")

    # N.2
    def test_ZDT2(self):
        function = functions.ZDT2(dimension=5)
        x = np.array([0, 0, 0, 0, 0])
        result = function(x)
        expected = [0, 1]
        self.assertTrue(np.allclose(result, expected), f"Expected {expected}, got {result}")

    # N.3
    def test_ZDT3(self):
        function = functions.ZDT3(dimension=5)
        x = np.array([0, 0, 0, 0, 0])
        result = function(x)
        expected = [0, 1]
        self.assertTrue(np.allclose(result, expected), f"Expected {expected}, got {result}")

    # N.4
    def test_ZDT4(self):
        function = functions.ZDT4(dimension=5)
        x = np.array([0, 0, 0, 0, 0])
        result = function(x)
        expected = [0, 1]
        self.assertTrue(np.allclose(result, expected), f"Expected {expected}, got {result}")

    # N.6
    def test_ZDT6(self):
        function = functions.ZDT6(dimension=5)
        x = np.array([0, 0, 0, 0, 0])
        result = function(x)
        expected = [1, 0]
        self.assertTrue(np.allclose(result, expected), f"Expected {expected}, got {result}")

    #Osyczka and Kundu
    def test_OsyczkaKundu(self):
        function = functions.OsyczkaKundu()
        x = np.array([1, 1, 3, 4, 5, 0])
        result = function(x)
        expected = [-46, 52]
        self.assertTrue(np.allclose(result, expected), f"Expected {expected}, got {result}")


    """
    =========================
    3 Objective Functions with 2 variables
    =========================
    """
    # Viennet Function
    def test_Viennet(self):
        function = functions.Viennet()
        x = np.array([0, 0])
        result = function(x)
        expected = [0, 17.03703703, -0.1]
        self.assertTrue(np.allclose(result, expected), f"Expected {expected}, got {result}")

if __name__ == '__main__':
    unittest.main()
