# app.py

import solara as sl
from mesa.visualization import SolaraViz, make_space_component
from model import WarehouseEnvModel
from agent import Shelf, DropZone

def agent_portrayal(agent):
    """
    Draw static objects:
      • Shelf → gray circle
      • DropZone → green circle
    """
    if isinstance(agent, Shelf):
        return {"color": "gray",  "r": 0.5}
    if isinstance(agent, DropZone):
        return {"color": "green", "r": 0.5}
    return {"color": "red",   "r": 0.3}

# Build the grid‐drawing component once
space = make_space_component(agent_portrayal)

@sl.component
def Page():
    # 1) Create a real model instance (so .grid definitely exists)
    model_inst = WarehouseEnvModel(width=20, height=10)

    # 2) Wrap it in Solara’s reactive system so the UI will pick up on any future changes
    reactive_model = sl.reactive(model_inst)

    # 3) Pass that instance *positionally* into SolaraViz
    return SolaraViz(
        reactive_model,   # <- your actual model instance
        [space],          # <- your grid component
        name="Warehouse Layout",
        play_interval=0   # <- static environment—no auto‑stepping
    )
