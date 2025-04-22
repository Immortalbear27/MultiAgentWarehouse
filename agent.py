# agent.py

from mesa import Agent

class Shelf(Agent):
    """
    A static shelf in the warehouse. No behavior.
    """
    def __init__(self, model):
        # Calls Mesa.Agent.__init__, setting self.model and self.unique_id
        super().__init__(model)

    def step(self):
        # Shelves do not act
        pass


class DropZone(Agent):
    """
    A static drop‑off zone in the warehouse. No behavior.
    """
    def __init__(self, model):
        super().__init__(model)

    def step(self):
        # Drop zones do not act
        pass
