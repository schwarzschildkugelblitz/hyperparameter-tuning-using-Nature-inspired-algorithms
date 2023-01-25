# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:50:25 2016

@author: hossam
"""
from asyncio.windows_events import NULL
from pathlib import Path
import optimizers.PSO as pso
# import optimizers.MVO as mvo
import optimizers.GWO as gwo
# import optimizers.GWO_copy as gwo_copy
import optimizers.MFO as mfo
import optimizers.CS as cs
import optimizers.BAT as bat
import optimizers.WOA as woa
import optimizers.FFA as ffa
import optimizers.SSA as ssa
import optimizers.GA as ga
import optimizers.HHO as hho
# import optimizers.HHOMP as hhomp
import optimizers.SCA as sca
import optimizers.JAYA as jaya
import optimizers.DE as de
# import optimizers.HHO_copy as hho_copy
# import optimizers.HHO_copy2 as hho_copy2
# import optimizers.GROM as grom
# import optimizers.MROM as mrom
# import optimizers.BBO as bbo
# import optimizers.CCO as cco
# import optimizers.COVIDHHO as covidhho
# import optimizers.MROM_SCA as mrom_sca
# import optimizers.TSA as tsa
# import optimizers.RLCO as rlco
# import optimizers.WOA_SCA_GWO as woa_sca_gwo
# import optimizers.SEO as seo 
# import optimizers.AHA as aha
# import optimizers.AHA_L as aha_l
import ML_Models
import csv
import numpy
import time
import warnings
import os
# import plot_convergence as conv_plot
# import plot_boxplot as box_plot
import Embedding
import numpy as np
from sklearn.model_selection import train_test_split

warnings.simplefilter(action="ignore")


def selector(algo, func_details, popSize, Iter, X_train, X_test, y_train, y_test):
    function_name = func_details[0]
    lb = func_details[1]
    ub = func_details[2]
    dim = func_details[3]

    if algo == "SSA":
        x = ssa.SSA(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "PSO":
        x = pso.PSO(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter,X_train, X_test, y_train, y_test)
    elif algo == "GA":
        x = ga.GA(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "BAT":
        x = bat.BAT(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "FFA":
        x = ffa.FFA(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "GWO":
        x = gwo.GWO(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter,X_train, X_test, y_train, y_test )
    # elif algo == "BBO":
    #     x = bbo.BBO(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "WOA":
        x = woa.WOA(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter,X_train, X_test, y_train, y_test )
    # elif algo == "MVO":
    #     x = mvo.MVO(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "MFO":
        x = mfo.MFO(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "CS":
        x = cs.CS(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "HHO":
        x = hho.HHO(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter,X_train, X_test, y_train, y_test )
    # elif algo == "HHOMP":
    #     x = hhomp.HHOMP(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "SCA":
        x = sca.SCA(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "JAYA":
        x = jaya.JAYA(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "DE":
        x = de.DE(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter,X_train, X_test, y_train, y_test )
    # elif algo == "HHO_copy":
    #     x = hho_copy.HHO(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    # elif algo == "HHO_copy2":
    #     x = hho_copy2.HHO_copy2(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    # elif algo == "GROM":
    #     x = grom.GROM(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    # elif algo == "MROM":
    #     x = mrom.MROM(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    # elif algo == "CCO":
    #     x = cco.CCO(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    # elif algo == "COVIDHHO":
    #     x = covidhho.COVIDHHO2(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    # elif algo == "GWO_copy":
    #     x = gwo_copy.GWO_copy(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    # elif algo == "MROM_SCA":
    #     x = mrom_sca.MROM_SCA(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    # elif algo == "TSA":
    #     x = tsa.TSA(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    # elif algo == "RLCO":
    #     x = rlco.RLCO(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    # elif algo == "WOA_SCA_GWO":
    #     x = woa_sca_gwo.WOA_SCA_GWO(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    # elif algo == "SEO":
    #     x = seo.SEO(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    elif algo == "AHA":
        x = aha.AHA(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    # elif algo == "AHA_L":
    #     x = aha_l.AHA_L(getattr(ML_Models, function_name), lb, ub, dim, popSize, Iter)
    else:
        return NULL
    return x


def run(optimizer, objectivefunc, NumOfRuns, params, export_flags, data_set):

    """
    It serves as the main interface of the framework for running the experiments.

    Parameters
    ----------
    optimizer : list
        The list of optimizers names
    objectivefunc : list
        The list of benchmark functions
    NumOfRuns : int
        The number of independent runs
    params  : set
        The set of parameters which are:
        1. Size of population (PopulationSize)
        2. The number of iterations (Iterations)
    export_flags : set
        The set of Boolean flags which are:
        1. Export (Exporting the results in a file)
        2. Export_details (Exporting the detailed results in files)
        3. Export_convergence (Exporting the covergence plots)
        4. Export_boxplot (Exporting the box plots)

    Returns
    -----------
    N/A
    """

    # Select general parameters for all optimizers (population size, number of iterations) ....
    PopulationSize = params["PopulationSize"]
    Iterations = params["Iterations"]

    # Export results ?
    # Export = export_flags["Export_avg"]
    Export_details = export_flags["Export_details"]
    # Export_convergence = export_flags["Export_convergence"]
    # Export_boxplot = export_flags["Export_boxplot"]

    Flag = False
    Flag_details = False

    # CSV Header for for the cinvergence
    CnvgHeader = []

    results_directory = time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    for l in range(0, Iterations):
        CnvgHeader.append("Iter_cross" + str(l + 1))
        CnvgHeader.append("Iter_acc" + str(l + 1))  
        CnvgHeader.append("Iter_recall" + str(l + 1))
        CnvgHeader.append("Iter_pre" + str(l + 1))
        CnvgHeader.append("Iter_f1" + str(l + 1))
    data_set_embbeding = Embedding.get_text_embeddings(data_set.iloc[:,0])
    X_train, X_test, y_train, y_test = train_test_split(data_set_embbeding,data_set.iloc[:,1], test_size=0.30, random_state=42)

    for i in range(0, len(optimizer)):
        for j in range(0, len(objectivefunc)):
            convergence = [0] * NumOfRuns
            executionTime = [0] * NumOfRuns
            for k in range(0, NumOfRuns):
                func_details = ML_Models.getFunctionDetails(objectivefunc[j])
                x = selector(optimizer[i], func_details, PopulationSize, Iterations,X_train, X_test, y_train, y_test)
                convergence[k] = x.convergence
                optimizerName = x.optimizer
                objfname = x.objfname
                if Export_details == True:
                    ExportToFile = results_directory + "experiment_details.csv"
                    with open(ExportToFile, "a", newline="\n") as out:
                        writer = csv.writer(out, delimiter=",")
                        if (
                            Flag_details == False
                        ):  # just one time to write the header of the CSV file
                            header = numpy.concatenate(
                                [["Optimizer", "objfname", "ExecutionTime"], CnvgHeader]
                            )
                            writer.writerow(header)
                            Flag_details = True  # at least one experiment
                        executionTime[k] = x.executionTime
                        a = numpy.concatenate(
                            [[x.optimizer, x.objfname, x.executionTime], np.array(x.convergence).flatten()]
                        )
                        writer.writerow(a)
                    out.close()

            # if Export == True:
            #     ExportToFile = results_directory + "experiment.csv"

            #     with open(ExportToFile, "a", newline="\n") as out:
            #         writer = csv.writer(out, delimiter=",")
            #         if (
            #             Flag == False
            #         ):  # just one time to write the header of the CSV file
            #             header = numpy.concatenate(
            #                 [["Optimizer", "objfname", "ExecutionTime"], CnvgHeader]
            #             )
            #             writer.writerow(header)
            #             Flag = True

            #         avgExecutionTime = float("%0.2f" % (sum(executionTime) / NumOfRuns))
            #         avgConvergence = numpy.around(
            #             numpy.mean(convergence, axis=0, dtype=numpy.float64), decimals=2
            #         ).tolist()
            #         a = numpy.concatenate(
            #             [[optimizerName, objfname, avgExecutionTime], avgConvergence]
            #         )
            #         writer.writerow(a)
            #     out.close()

    # if Export_convergence == True:
    #     print(optimizer)
    #     conv_plot.run(results_directory, optimizer, objectivefunc, Iterations)

    # if Export_boxplot == True:
    #     box_plot.run(results_directory, optimizer, objectivefunc, Iterations)

    if Flag == False:  # Faild to run at least one experiment
        print(
            "No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions"
        )

    print("Execution completed")
