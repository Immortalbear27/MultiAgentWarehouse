# agent.py

from mesa import Agent

class Shelf(Agent):
    # Using pass as it is a static agent - Doesn't require anything else
    pass

class DropZone(Agent):
    # Using pass as it is a static agent - Doesn't require anything else
    pass

class WarehouseAgent(Agent):
    """
    A simple warehouse robot: picks a random neighbor each step.
    """
    def __init__(self, model):
        super().__init__(model)
        self.state = "idle"
        self.path  = []
        self.task_steps = 0

    def step(self):
        # If no path to follow or just delivered, skip movement
        if not self.path or getattr(self, 'state', None) == 'to_dropoff' and not self.path:
            return

        next_cell = self.path[0]
        # Collision anticipation
        if self._will_collide(next_cell):
            self.model.collisions += 1
            self._replan()
            return

        # Move along path
        self.model.grid.move_agent(self, next_cell)
        self.path.pop(0)
        self.task_steps += 1

    def _will_collide(self, cell):
        """
        Return True if another WarehouseAgent occupies the target cell.
        """
        return any(
            isinstance(a, WarehouseAgent) and a is not self
            for a in self.model.grid.get_cell_list_contents([cell])
        )

    def _replan(self):
        """
        Recompute path around the current target.
        """
        goal = self.pickup_pos if self.state == 'to_pickup' else self.next_drop
        new_path = self.model.compute_path(self.pos, goal)
        if new_path:
            self.path = new_path


class ShelfItem(Agent):
    # Static agent once again, so no additional logic required
    pass
