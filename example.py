import nevergrad as ng
import matplotlib.pyplot as plt
from nevergrad.functions.multiobjective.functions import (
BinhKorn, ChankongHaimes, Kursawe, FonsecaFleming, TestFunction4, Viennet, Schaffer1, Schaffer2, 
CTP1, ConstrEx, OsyczkaKundu, ZDT1, ZDT2, ZDT3, ZDT6, ZDT4
)

def main():
    # Initialize the BinhKorn multi-objective function (Feel free to change the objective functions in our functions.py files)
    function = BinhKorn()

    # Set up the optimizer with Differential Evolution
    optimizer = ng.optimizers.DE(parametrization=function.parametrization, budget=500)
    
    # Run the optimization process
    for _ in range(optimizer.budget):
        x = optimizer.ask()
        loss = function(x.value)
        optimizer.tell(x, loss)

    # Obtain and print the recommended solution
    recommendation = optimizer.provide_recommendation()
    print("Recommended solution:", recommendation.value)
    print("Objective values:", function(recommendation.value))

    # Visualize the Pareto front obtained from the optimization
    pareto_front = optimizer.pareto_front()
    f1_values = [p.losses[0] for p in pareto_front]
    f2_values = [p.losses[1] for p in pareto_front]
    plt.scatter(f1_values, f2_values)
    plt.xlabel("f1(x)")
    plt.ylabel("f2(x)")
    plt.title(f"Pareto front for {function.name}")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
