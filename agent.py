# agent.py

from mesa import Agent

class Shelf(Agent):
    def __init__(self, model):
        super().__init__(model)
    def step(self):
        pass  # static

class DropZone(Agent):
    def __init__(self, model):
        super().__init__(model)
    def step(self):
        pass  # static

class WarehouseAgent(Agent):
    """
    A simple warehouse robot: picks a random neighbor each step.
    """
    def __init__(self, model):
        super().__init__(model)
    def step(self):
        # Get non‐diagonal neighbors
        neighbours = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False
        )
        # Choose one at random and move there
        new_pos = self.random.choice(neighbours)
        self.model.grid.move_agent(self, new_pos)
