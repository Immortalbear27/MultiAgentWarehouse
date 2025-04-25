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
        self.steps_taken = 0
        self.task_steps = 0

    def step(self):
        if self.path:
            next_cell = self.path.pop(0)
            self.model.grid.move_agent(self, next_cell)
            # 1️⃣ energy use: count your move
            self.task_steps += 1
        # 2️⃣ detect delivery completion
        elif getattr(self, "state", None) == "to_dropoff":
            # when model.step() sees path empty again, it flips state→"idle"
            # so here you could also signal:
            # self.model.total_deliveries += 1
            pass


class ShelfItem(Agent):
    """
    A single item sitting on a shelf. Static.
    """
    def __init__(self, model):
        super().__init__(model)
    def step(self):
        pass
