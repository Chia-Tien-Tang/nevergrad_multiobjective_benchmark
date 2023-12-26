import numpy as np
from nevergrad.parametrization import parameter as p

class ConstrainedMultiObjective:
    """
    A base class for defining multi-objective functions with constraints.

    Attributes:
        name (str): Name of the multi-objective function.
        parametrization (p.Array): Parameter space for the optimization problem.
    """

    def __init__(self, name: str, parametrization: p.Array):
        """
        Initializes the multi-objective function with a name and parametrization.

        Args:
            name (str): Name of the function.
            parametrization (p.Array): Parameter space for the function.
        """
        self.name = name
        self.parametrization = parametrization

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        This method should be implemented in subclasses to calculate the objective values.

        Args:
            x (np.ndarray): An array of parameter values.

        Returns:
            np.ndarray: An array of objective values.
        """
        pass # Implementation should be provided in subclasses