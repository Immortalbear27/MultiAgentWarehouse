# app.py

import solara as sl
from mesa.visualization import SolaraViz, make_space_component, make_plot_component, Slider
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
        color = "black"
        size = 60
    elif agent_type == "DropZone":
        color = "green"
        size = 60
    elif agent_type == "WarehouseAgent":
        color = "blue"
        size = 60
    elif agent_type == "ShelfItem":
        color = "orange"
        size = 10
    else:
        color = "red"  # unexpected proxy type
        size = 60
    return {"color": color, "size": size}

# Build the grid‐drawing component once
space = make_space_component(agent_portrayal)

# Create a plot component for your metrics
plot = make_plot_component(
    ["Throughput", "Collisions", "PendingTasks", "AvgStepsPerDelivery"],
    backend="matplotlib"
)

@sl.component
def Page():
    """
    - Creates a reactive model instance so .grid is always present.
    - Renders static shelves/drop‑zones + one moving robot.
    - Use Reset/Step/Play to see the blue WarehouseAgent wander.
    """
    # 1️⃣ Create the model instance with fixed dims
    model_inst = WarehouseEnvModel(width=30, 
                                   height=25,
                                   shelf_edge_gap = 2,
                                   aisle_interval = 5, 
                                   num_agents = 8)

    # 2️⃣ Wrap it in Solara’s reactive system
    reactive_model = sl.reactive(model_inst)
    
    # Slider Logic:
    model_params = {
        "num_agents": Slider("Number of Agents", 8, 1, 20, step = 1),
        "width": Slider("Width of Warehouse", 30, 10, 60),
        "height": Slider("Height of Warehouse", 25, 10, 60),
        "strategy": {
            "type": "Select",
            "value": "centralised",
            "values": ["centralised", "decentralised", "swarm"],
            "label": "Coordination Strategy"
        }
    }

    # 3️⃣ Pass it positionally to SolaraViz along with your space drawer
    return SolaraViz(
        reactive_model,  # must be an instance so .grid exists
        [space, plot],         # your grid component
        model_params = model_params,
        name="Warehouse Layout",
        play_interval=500  # ms between steps
    )
