import solara as sl
from mesa.visualization import SolaraViz, make_space_component, make_plot_component, Slider
from model import WarehouseEnvModel

def agent_portrayal(agent):
    """
    Determine visualisation settings for each type of agent in the GUI.
    
    Colour‐code each agent type:
      - Shelf -> Black
      - DropZone -> Purple
      - WarehouseAgent -> Green
      - ShelfItem -> Orange
    """
    agent_type = agent.__class__.__name__
    if agent_type == "Shelf":
        colour = "black"
        size = 60
    elif agent_type == "DropZone":
        colour = "purple"
        size = 150
    elif agent_type == "WarehouseAgent":
        colour = "green"
        size = 60
    elif agent_type == "ShelfItem":
        colour = "orange"
        size = 10
    else:
        colour = "red"
        size = 60
    return {"color": colour, "size": size}

# Build the grid‐drawing component once:
space = make_space_component(agent_portrayal)

# Create a plot component for the metrics:
plot = make_plot_component(
    ["total_deliveries", "collisions", "pendingtasks", "AvgStepsPerDelivery"],
    backend="matplotlib"
)

@sl.component
def ExportButton(model):
    """
    A simple button that, when clicked, will pull the current
    DataCollector dataframe off the model and write it out as CSV.
    """
    sl.Button(
        "Export CSV",
        on_click=lambda: model.datacollector
                             .get_model_vars_dataframe()
                             .to_csv("results.csv", index=False),
    )


@sl.component
def Page():
    """
    - Creates a reactive model instance so .grid is always present.
    - Renders static shelves/drop‑zones + one moving robot.
    - Use Reset/Step/Play to see the blue WarehouseAgent wander.
    """
    # Create the model instance with various settings:
    model_inst = WarehouseEnvModel(width=20, 
                                   height=15,
                                   shelf_edge_gap = 2,
                                   aisle_interval = 6, 
                                   num_agents = 5,
                                   drop_zone_size = 2,
                                   max_steps = 1000,
                                   search_radius = 3)

    # Wrap it in Solara’s reactive system:
    reactive_model = sl.reactive(model_inst)
    
    # Slider Logic:
    model_params = {
        "num_agents": Slider("Number of Agents", 5, 1, 20, step = 1),
        "width": Slider("Width of Warehouse", 20, 10, 60),
        "height": Slider("Height of Warehouse", 15, 10, 60),
        "shelf_edge_gap": Slider("Shelf Edge Gap", 2, 1, 5, step = 1),
        "aisle_interval": Slider("Aisle Interval", 6, 2, 10, step = 1),
        "drop_zone_size": Slider("Size of Drop Zones", 2, 1, 3, step = 1),
        "item_respawn_delay": Slider("Item Respawn Delay", 50, 0, 200, step = 1),
        "strategy": {
            "type": "Select",
            "value": "centralised",
            "values": ["centralised", "decentralised", "swarm"],
            "label": "Coordination Strategy"
        },
        "respawn_enabled": {
            "type": "Checkbox",
            "value": "True",
            "label": "Enable Item Respawn?"
        }
    }

    # Pass it to SolaraViz along with the space drawer:
    return SolaraViz(
        reactive_model, 
        [space, plot, ExportButton],
        model_params = model_params,
        name="Warehouse Layout",
        play_interval=25
    )

