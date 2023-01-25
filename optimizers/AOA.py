"""
Code by Harshit Batra
"""
import random
import numpy
import math
from solution import solution
import time


def HHO(objf, lb, ub, dim, SearchAgents_no, Max_iter):

    # dim=30
    # SearchAgents_no=50
    # lb=-100
    # ub=100
    # Max_iter=500
    EPSILON = 10E-10
    alpha = 5
    miu = 0.5
    Min = 0.2
    Max = 0.9
    # initialize the location and Energy of the rabbit
    best_Location = numpy.zeros(dim)
    best_Fitness = float("inf")  # change this to -inf for maximization problems

    if not isinstance(lb, list):
        lb = [lb for _ in range(dim)]
        ub = [ub for _ in range(dim)]
    lb = numpy.asarray(lb)
    ub = numpy.asarray(ub)

    # Initialize the locations of Harris' hawks
    X = numpy.asarray(
        [x * (ub - lb) + lb for x in numpy.random.uniform(0, 1, (SearchAgents_no, dim))]
    )

    # Initialize convergence
    convergence_curve = numpy.zeros(Max_iter)

    ############################
    s = solution()

    print('HHO is now tackling  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################

    t = 0  # Loop counter

    # Main loop
    while t < Max_iter:
        for i in range(0, SearchAgents_no):

            # Check boundries

            X[i, :] = numpy.clip(X[i, :], lb, ub)

            # fitness of locations
            fitness = objf(X[i, :])

            # Update the location of Rabbit
            if fitness < best_Fitness :  # Change this to > for maximization problem
                best_Fitness = fitness
                best_Location = X[i, :].copy()

        Moa = Min + t*((Max - Min)/Max_iter)
        Mop = 1 - ((t) ** (1.0 / alpha)) / (t ** (1.0 / alpha))

        E1 = 2 * (1 - (t / Max_iter))  # factor to show the decreaing energy of rabbit

        # Update the location of Harris' hawks
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):
                r1, r2, r3 = numpy.random.rand(3)
                if r1 > Moa:
                    if r2 < 0.5:
                        X[i, j] = best_Location[j] / (Mop + EPSILON) * ((ub[j] - lb[j]) * miu + lb[j])
                    else:
                        X[i, j] = best_Location[j] *(Mop) * ((ub[j] - lb[j]) * miu + lb[j])

                else:
                    if r3 < 0.5:
                        best_Location[j] - (Mop) * ((ub[j] - lb[j]) * miu + lb[j])
                    else:
                        best_Location[j] + (Mop) * ((ub[j] - lb[j]) * miu + lb[j])


        convergence_curve[t] = best_Location
        if t % 1 == 0:
            print(
                [
                    "At iteration "
                    + str(t)
                    + " the best fitness is "
                    + str(best_Fitness)
                ]
            )
        t = t + 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "HHO"
    s.objfname = objf.__name__
    s.best = best_Fitness
    s.bestIndividual = best_Location

    return s


def Levy(dim):
    beta = 1.5
    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = 0.01 * numpy.random.randn(dim) * sigma
    v = numpy.random.randn(dim)
    zz = numpy.power(numpy.absolute(v), (1 / beta))
    step = numpy.divide(u, zz)
    return step

