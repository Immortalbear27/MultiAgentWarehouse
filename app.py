# app.py

import solara as sl
from mesa.visualization import SolaraViz, make_space_component
from model import WarehouseEnvModel
from agent import Shelf, DropZone, WarehouseAgent

def agent_portrayal(agent):
    """
    Colour‐code each agent type:
      • Shelf          → gray
      • DropZone       → green
      • WarehouseAgent → blue
    """
    agent_type = agent.__class__.__name__
    if agent_type == "Shelf":
        color = "gray"
    elif agent_type == "DropZone":
        color = "green"
    elif agent_type == "WarehouseAgent":
        color = "blue"
    else:
        color = "red"  # unexpected proxy type
    return {"color": color, "r": 0.5}

# Build the grid‐drawing component once
space = make_space_component(agent_portrayal)

@sl.component
def Page():
    """
    - Creates a reactive model instance so .grid is always present.
    - Renders static shelves/drop‑zones + one moving robot.
    - Use Reset/Step/Play to see the blue WarehouseAgent wander.
    """
    # 1️⃣ Create the model instance with fixed dims
    model_inst = WarehouseEnvModel(width=20, height=10)

    # 2️⃣ Wrap it in Solara’s reactive system
    reactive_model = sl.reactive(model_inst)

    # 3️⃣ Pass it positionally to SolaraViz along with your space drawer
    return SolaraViz(
        reactive_model,  # must be an instance so .grid exists
        [space],         # your grid component
        name="Warehouse Layout",
        play_interval=500  # ms between steps
    )
