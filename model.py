# model.py

from mesa import Model
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from agent import Shelf, DropZone, WarehouseAgent

class WarehouseEnvModel(Model):
    """
    Warehouse with static shelves/drop‐zones + one moving agent.
    """
    def __init__(
        self,
        width: int = 20,
        height: int = 10,
        shelf_coords: list[tuple[int,int]] = None,
        drop_coords:  list[tuple[int,int]] = None,
        seed:         int = None
    ):
        super().__init__(seed=seed)
        self.grid     = MultiGrid(width, height, torus=False)
        self.schedule = SimultaneousActivation(self)

        # Default static layout
        if shelf_coords is None:
            mid = height // 2
            shelf_coords = [(x, mid) for x in range(width // 5, width * 4 // 5)]
        if drop_coords is None:
            drop_coords = [(0, 0), (width - 1, height - 1)]

        # Place static agents
        for pos in shelf_coords:
            s = Shelf(self)
            self.grid.place_agent(s, pos)
        for pos in drop_coords:
            dz = DropZone(self)
            self.grid.place_agent(dz, pos)

        # Place one WarehouseAgent in a random empty cell
        robot = WarehouseAgent(self)
        self.schedule.add(robot)
        x, y = self.random.randrange(width), self.random.randrange(height)
        self.grid.place_agent(robot, (x, y))

    def step(self):
        """
        Advance all scheduled agents (only the robot actually moves).
        """
        self.schedule.step()
