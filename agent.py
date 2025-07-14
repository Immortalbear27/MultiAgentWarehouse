# Imports:
from mesa import Agent

class Shelf(Agent):
    # Using pass as it is a static agent - Doesn't require anything else
    pass

class DropZone(Agent):
    # Using pass as it is a static agent - Doesn't require anything else
    pass

class ShelfItem(Agent):
    # Static agent once again, so no additional logic required
    pass

class WarehouseAgent(Agent):
    """
    Mobile agent representing a warehouse robot that picks up and delivers items.

    Each agent maintains its state machine transitioning through:
      - 'idle': waiting for task
      - 'to_pickup': moving to pickup location
      - 'to_dropoff': moving to dropoff location
      - 'relocating': moving to a staging area after dropoff

    Tracks path following, step counting, delivery counts, and deposits pheromone trails.
    """
    
    def __init__(self, model):
        """
        Initialize a new WarehouseAgent.

        Args: Reference to the simulation model instance.
        """
        super().__init__(model)
        self.state = "idle" # Initial state
        self.path  = [] # Planned sequence of (x, y) steps
        self.task_steps = 0 # Steps taken on current tasks
        self.deliveries = 0 # Total completed deliveries

    def step(self):
        """
        Perform one movement step or handle collisions and replanning.

        If the agent has a non-empty path and is not completing a dropoff,
        it attempts to move along its path, handling collisions or depositing
        pheromones after movement.
        """
        
        # Skip movement if no path or if just completed dropoff and no path remains:
        if not self.path or getattr(self, 'state', None) == 'to_dropoff' and not self.path:
            return

        # Next desired position:
        next_cell = self.path[0]
        
        # Check for collision at next cell:
        if self._will_collide(next_cell):
            # Record collision event and attempt to replan route:
            self.model.collisions += 1
            self._replan()
            return

        # Move agent on the grid and consume the first path step:
        self.model.grid.move_agent(self, next_cell)
        self.path.pop(0)
        # Increment the step count for active task
        self.task_steps += 1
        
         # Deposit pheromone at new position to influence other agents:
        self.model.pheromones[self.pos] += self.model.pheromone_deposit

    def _will_collide(self, cell):
        """
        Check if another WarehouseAgent currently occupies the target cell.

        Args: Grid coordinate to test for collision.

        Returns: True if a different WarehouseAgent is present at `cell`, False otherwise.
        """
        # Inspect the contents of the target cell for other agents:
        return any(
            isinstance(a, WarehouseAgent) and a is not self
            for a in self.model.grid.get_cell_list_contents([cell])
        )

    def _replan(self):
        """
        Recompute the agent’s path to its current goal, 
        clearing and updating the cache.

        Uses reservation-aware A* or greedy compute_path_to_drop 
        when en-route to dropoff.
        """
        
        # Determine goal based on current state:
        goal = self.pickup_pos if self.state == 'to_pickup' else self.next_drop
        
        # Remove any cached path for current position, goal:
        cache_key = (self.pos, goal)
        if cache_key in self.model.path_cache:
            del self.model.path_cache[cache_key]
        new_path = self.model.compute_path(self.pos, goal)
        
        # Choose pathfinding method - Greedy for drop-off, A* otherwise:
        if getattr(self, "state", None) == "to_dropoff" and goal in self.model.drop_coords:
            self.path.clear()
            new_path = self.model.compute_path_to_drop(self.pos, goal)
        else:
            self.path.clear()
            new_path = self.model.compute_path(self.pos, goal)
            
        # Replace the current path with the newly computed one:
        self.path.extend(new_path)
