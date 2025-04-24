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
        self.state = "idle"
        self.path  = []

    def step(self):
        if getattr(self, "path", None):
            next_cell = self.path.pop(0)
            self.model.grid.move_agent(self, next_cell)


class ShelfItem(Agent):
    """
    A single item sitting on a shelf. Static.
    """
    def __init__(self, model):
        super().__init__(model)
    def step(self):
        pass
