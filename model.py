# model.py

from mesa import Model
from mesa.space import MultiGrid
from agent import Shelf, DropZone

class WarehouseEnvModel(Model):
    """
    Warehouse environment with static shelves and drop zones.
    """
    def __init__(
        self,
        width: int = 20,
        height: int = 10,
        shelf_coords: list[tuple[int, int]] = None,
        drop_coords: list[tuple[int, int]] = None,
        seed: int = None
    ):
        # Initialize Mesa base (sets self.random, step counters, etc.)
        super().__init__(seed=seed)

        # Create a 2D grid
        self.grid = MultiGrid(width, height, torus=False)

        # Default shelf positions: middle horizontal band
        if shelf_coords is None:
            mid = height // 2
            shelf_coords = [(x, mid) for x in range(width // 5, width * 4 // 5)]

        # Default drop‑off zones: top‑left and bottom‑right corners
        if drop_coords is None:
            drop_coords = [(0, 0), (width - 1, height - 1)]

        # Place Shelf agents
        for pos in shelf_coords:
            agent = Shelf(self)
            self.grid.place_agent(agent, pos)

        # Place DropZone agents
        for pos in drop_coords:
            agent = DropZone(self)
            self.grid.place_agent(agent, pos)

    def step(self):
        # No dynamics yet
        return
