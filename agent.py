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
            next_cell = self.path[0]

            # ——— Collision *anticipation* detection ——————————
            occupants = self.model.grid.get_cell_list_contents([next_cell])
            # anyone else about to occupy that cell?
            blockers = [a for a in occupants if isinstance(a, WarehouseAgent) and a is not self]
            if blockers:
                # Count this as a collision attempt
                self.model.collisions += 1

                # (Optionally) only one of the colliding pair should increment:
                # if self.unique_id < blockers[0].unique_id:
                #     self.model.collisions += 1

                # Now replan around the blocker
                goal = (self.pickup_pos if self.state=="to_pickup" 
                        else self.next_drop)
                new_path = self.model.compute_path(self.pos, goal)
                if new_path:
                    self.path = new_path
                return  # skip moving this tick

            # ——— No blocker, so actually move ——————————
            self.path.pop(0)
            self.model.grid.move_agent(self, next_cell)
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
