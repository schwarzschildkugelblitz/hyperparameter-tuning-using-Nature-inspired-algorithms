from optimizer import run
import pandas as pd 

optimzer = ["PSO"]
objective_fuction = ["LSTM_"]
num_run = 1 
params ={"PopulationSize":3, "Iterations":5}
exportflags={
    "Export_details": True,
}
df = pd.read_csv("data_set/Musical_instruments_reviews.csv",header=None)
run(optimzer,objective_fuction,num_run,params,exportflags,df)
