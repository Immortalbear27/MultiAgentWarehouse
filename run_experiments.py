from mesa.batchrunner import BatchRunner
from model import WarehouseEnvModel

variable_params = {
    "strategy":       ["centralised","decentralised","swarm"],
    "num_agents":     [5,10,15,20],
    "shelf_edge_gap": [1,2,3],
    "aisle_interval": [5,10],
}

batch = BatchRunner(
    WarehouseEnvModel,
    variable_params,
    {},
    iterations=30,
    max_steps=1000,
    model_reporters=WarehouseEnvModel.datacollector.model_reporters
)
batch.run_all()
results = batch.get_model_vars_dataframe()
results.to_csv("results.csv", index=False)
