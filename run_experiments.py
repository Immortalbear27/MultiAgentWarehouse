from mesa.batchrunner import batch_run
from model import WarehouseEnvModel
import pandas as pd

variable_params = {
    "strategy":       ["centralised","decentralised","swarm"],
    "num_agents":     [5,10,15,20],
    "shelf_edge_gap": [1,2,3],
    "aisle_interval": [5,10],
    "shelf_rows": [3, 5, 7],
    "width": [20, 30, 40],
    "height": [20, 30, 40],
    "search_radius": [1, 3, 5]
}

if __name__ == "__main__":
    results = batch_run(
        WarehouseEnvModel,
        parameters = variable_params,
        iterations=30,
        max_steps=500,
        data_collection_period = 1,
        number_processes = None,
        display_progress = True
    )

    df = pd.DataFrame(results)
    df.to_csv("batch_results.csv", index = False)
