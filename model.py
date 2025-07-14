# Imports:
from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from agent import Shelf, DropZone, WarehouseAgent, ShelfItem
from collections import deque
import heapq
from math import inf
from scipy.optimize import linear_sum_assignment
from numba import njit
import numpy as np

   
@njit
def best_adj_for_pickup(ax, ay, px, py,
                        neigh_arr, neigh_len,
                        heatmap, pheromone,
                        deliveries, alpha, beta, gamma):
    """
    For one agent (ax,ay) & one pickup (px,py), scan its free adjacents:
      neigh_arr[j, k] = the k’th neighbor for pickup j,
      neigh_len[j]    = # valid neighbors for pickup j.
    Returns (best_cost, best_k).
    """
    best_cost = 1e18
    best_k    = 0
    # j index is carried in the outer loop
    # here we assume the caller slices neigh_arr[j], neigh_len[j]
    for k in range(neigh_len):
        nx = neigh_arr[k, 0]
        ny = neigh_arr[k, 1]
        # simple Manhattan dist to entry
        d = abs(ax - nx) + abs(ay - ny)
        # no per‐path heat summation here—too expensive
        cost = d + beta * deliveries - gamma * pheromone[nx, ny]
        if cost < best_cost:
            best_cost = cost
            best_k    = k
    return best_cost, best_k

@njit
def make_cost_matrix(agent_pos, task_pos,
                     neigh_arr_3d, neigh_lens,
                     heatmap, pheromone,
                     deliveries, alpha, beta, gamma):
    """
    Build (n_agents × n_tasks) cost matrix & best‐entry indices.
    neigh_arr_3d: shape (n_tasks, maxL, 2)
    neigh_lens:    shape (n_tasks,)
    heatmap, pheromone: shape (width, height)
    deliveries:    shape (n_agents,)
    """
    n_agents = agent_pos.shape[0]
    n_tasks  = task_pos.shape[0]
    cost_mat = np.empty((n_agents, n_tasks), np.float64)
    best_k   = np.empty((n_agents, n_tasks), np.int64)
    for i in range(n_agents):
        ax, ay = agent_pos[i, 0], agent_pos[i, 1]
        deliv   = deliveries[i]
        for j in range(n_tasks):
            neigh_arr = neigh_arr_3d[j]
            length    = neigh_lens[j]
            c, k = best_adj_for_pickup(
                ax, ay,
                task_pos[j,0], task_pos[j,1],
                neigh_arr, length,
                heatmap, pheromone,
                deliv, alpha, beta, gamma
            )
            cost_mat[i, j]  = c
            best_k[i, j]    = k
    return cost_mat, best_k


class WarehouseEnvModel(Model):
    """
    Warehouse with static shelves/drop‐zones + one moving agent.
    """
    def __init__(
        self,
        width = 30,
        height = 30,
        drop_coords = None,
        drop_zone_size = 1,
        shelf_rows = 5,
        shelf_edge_gap = 1,
        aisle_interval = 5,
        num_agents = 3,
        strategy = "centralised",
        item_respawn_delay = 50,
        respawn_enabled = True,
        max_steps = 500,
        search_radius = 3,
        auction_radius = 10,
        seed = None
    ):
        """
        Initialize the warehouse environment model.

        Parameters:
            width: Number of columns in the grid.
            height: Number of rows in the grid.
            drop_coords: Custom drop-zone coordinates.
            drop_zone_size Side length of each drop-zone square.
            shelf_rows: Number of horizontal shelf rows.
            shelf_edge_gap: Gap between shelves and grid edges.
            aisle_interval: Column spacing between aisles.
            num_agents: Number of WarehouseAgent instances to spawn.
            strategy: Task-assignment strategy ('centralised', 'decentralised', 'swarm').
            item_respawn_delay: Delay (in ticks) before an item reappears.
            respawn_enabled: Whether to re-spawn shelf items.
            max_steps: Maximum number of steps before stopping.
            search_radius: Bounding-box radius for pathfinding.
            auction_radius: Max distance for bidding in decentralised strategy.
            seed: Random seed for reproducibility.
        """
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.strategy = strategy
        self.num_agents = num_agents
        self.drop_zone_size = drop_zone_size
        self.shelf_edge_gap = shelf_edge_gap
        self.aisle_interval = aisle_interval
        self.item_respawn_delay = item_respawn_delay
        self.respawn_enabled = respawn_enabled
        self.search_radius = search_radius
        self.auction_radius = auction_radius
        self.respawn_queue: deque[tuple[tuple[int, int], int]] = deque()
        self.max_steps = max_steps
        self.path_cache: dict[tuple[tuple[int,int],tuple[int,int]], list[tuple[int,int]]] = {}
        self.reservations: dict[tuple[int,int,int], int] = {}
        
        # Initialise the grid and scheduler:
        self.grid     = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        
        # Pre-compute 4-way neighbours for each cell:
        self.neighbours = {
            (x, y): self.grid.get_neighborhood((x, y), moore = False, include_center = False)
            for x in range(width) for y in range(height)
        }
        
        # Initialise the heatmap:
        self.heatmap = {
            (x, y): 0
            for x in range(self.width)
            for y in range(self.height)
        }
        
        # Pheromone Parameters
        # Initialise pheromones:
        self.pheromones = {
            (x, y): 0.0
            for x in range(self.width)
            for y in range(self.height)
        }
        self.pheromone_deposit  = 1.0   # How much each robot leaves per tick
        self.pheromone_evap_rate = 0.05 # Fraction to evaporate each tick
        self.gamma = 0.5  # Weight for pheromone attraction in cost
        
        # Assign pheromone field - Efficiency Increase:
        self.pheromone_field = np.zeros((self.width, self.height), dtype = np.float64)

        
        # Counters for metrics:
        self.total_deliveries = 0 # Amount of total deliveries made
        self.collisions = 0 # Counter for collisions
        self.congestion_accum = 0 # Cumulative congestion counter
        self.ticks = 0 # Used for averaging
        self.total_task_steps = 0 # Counter for total steps taken for tasks
        
        # DataCollector reporters:
        self.datacollector = DataCollector(
            model_reporters={
                "total_deliveries":      lambda m: m.total_deliveries,
                "ticks": lambda m: m.ticks,
                "AvgStepsPerDelivery":(
                    lambda m: m.total_task_steps / m.total_deliveries
                    if m.total_deliveries > 0 else 0
                ),
                "collisions":      lambda m: m.collisions,
                "pendingtasks":    lambda m: len(m.tasks),
                "AvgCongestion": (lambda m: m.congestion_accum / m.ticks 
                                  if m.ticks > 0 else 0),
                "total_task_steps": lambda m: m.total_task_steps
            }
        )

        # Compute the Y-positions of each shelf row:
        row_positions = [
            int(round((i+1) * height / (shelf_rows+1)))
            for i in range(shelf_rows)
        ]
        
        # Initialise the shelf agents:
        self.shelf_coords = self.create_shelves(row_positions, self.shelf_edge_gap, self.aisle_interval)
        
        # Initialise the drop zone agents:
        self.drop_coords = self.create_drop_zones(drop_coords)
        
        # Compute the full grid for more efficient path finding calculations later on:
        self.compute_drop_zone_distances()
            
        # Initialize items: 1 item per shelf cell:        
        self.items, self.item_agents = self.create_items(self.shelf_coords)

        # Build initial task list - One task per item, with drop randomly chosen:
        self.tasks = self.create_tasks()

        # Shuffle tasks so they are in random order:
        self.random.shuffle(self.tasks)

        # Spawn multiple robots:
        self.robots = self.spawn_robots(self.num_agents)

    def compute_drop_zone_distances(self):
        """
        Computes shortest grid‐distances from every cell to the nearest drop-zone.

        Uses a multi-source BFS seeded at all drop-zone coordinates,
        propagating outwards over the 4-connected grid. Treats shelves as impassable
        obstacles but ignores robots and reservation markers.
        """
        self.static_distance = {
            (x, y): inf
            for x in range(self.width)
            for y in range(self.height)
        }
        queue = deque()

        # Seed the queue with drop-zone cells:
        for dropzone in self.drop_coords:
            self.static_distance[dropzone] = 0
            queue.append(dropzone)

        # 4-way grid BFS:
        while queue:
            x, y = queue.popleft()
            d0   = self.static_distance[(x, y)]
            for nx, ny in self.neighbours[(x, y)]:
                # Skip shelves (static obstacles):
                if any(isinstance(o, Shelf) for o in self.grid.get_cell_list_contents([(nx, ny)])):
                    continue
                if self.static_distance[(nx, ny)] > d0 + 1:
                    self.static_distance[(nx, ny)] = d0 + 1
                    queue.append((nx, ny))
                    
    def compute_path_to_drop(self, start, goal):
        """
        Compute a reservation-aware path from start to a drop-zone.

        Uses a greedy descent at each step, moving to an adjacent cell with 
        strictly lower static_distance that is passable and not reserved at 
        the next timestep to build a path. If no valid move exists, or if the 
        search loops beyond the limit, it falls back to full A* via compute_path().

        Args:
            start: Starting (x, y) coordinate.
            goal: Target (x, y) coordinate.

        Returns: Sequence of grid coordinates from start to goal 
        (exclusive of start). Returns an empty list if no path is 
        found, or the result of compute_path().
        """
        # Simulation Time:
        now = self.schedule.time

        # If goal isn't a drop-zone, then skip greedy pathfinding and use A* search:
        if goal not in self.drop_coords:
            return self.compute_path(start, goal)

        # Variable Instantiations
        # path - List to accumulate chosen steps.
        # x0, y0 - Outlines current position.
        # time - As the moves are conducted, time keeps a track of when moves are made.
        # loop-count - Counter to detect infinite loops.
        # max_loops - Safety measure, prevents massive expansions in pathfinding.
        path = []
        x0, y0 = start
        time = now
        loop_count = 0
        max_loops = self.width * self.height * 4

        # Continue execution until goal coordinate is met:
        while (x0, y0) != goal:
            loop_count += 1
            
            # Safety measure - if no path found within a certain amount of loops, then give up:
            if loop_count > max_loops:
                return []

            # Get the neighbouring cells:
            local_neighbours = self.neighbours[(x0, y0)]
            
            # Filter for valid moves with three requirements:
            # - Must move to cell closer to drop-zone
            # - That cell must not be reserved by another agent at the next timestep
            # - Cell must be passable i.e. not a shelf or out of the warehouse bounds
            candidates = [
                (nx, ny) for nx, ny in local_neighbours
                if self.static_distance.get((nx, ny), inf) < self.static_distance[(x0, y0)]
                and not isinstance(self.reservations.get((nx, ny, time+1)), int)
                and self.is_passable((nx, ny), goal)
            ]
            
            # If no greedy move is possible, default to A* search:
            if not candidates:
                return self.compute_path(start, goal)

            # Pick the neighbour that most decreases the distance to the drop-zone:
            x1, y1 = min(candidates, key=lambda c: self.static_distance[c])
            path.append((x1, y1))
            
            # Reserve cell at the next time-step to avoid collisions.
            # Use None as placeholder reservation - Treated as non-blocking in A*:
            self.reservations[(x1, y1, time+1)] = None
            
            # Update position and time for the next iteration:
            x0, y0, time = x1, y1, time + 1
        return path

    def spawn_robots(self, num_agents: int) -> list[WarehouseAgent]:
        """
        Instantiate and deploy a group of warehouse robots.

        Creates `num_agents` new WarehouseAgent objects, registers each
        with the scheduler, places it on a randomly chosen empty grid cell,
        and returns the list of all created agents.

        Args:
            num_agents: Number of agents to spawn.

        Returns: The newly created and placed agents.
        """
        # Collects the agents that will be scheduled to spawn:
        robots: list[WarehouseAgent] = []
        
        for _ in range(num_agents):
            # Instantiate a new agent:
            robot = WarehouseAgent(self)
            # Register with the scheduler so it's activated each step:
            self.schedule.add(robot)
            # Pick an unoccupied grid cell at random:
            x, y = self.random_empty_cell()
            # Place the agent onto the grid at (x, y):
            self.grid.place_agent(robot, (x, y))
            # Keep track of it within the list of robots:
            robots.append(robot)
        return robots

    def create_tasks(self) -> list[tuple[int,int]]:
        """
        Generate a shuffled list of tasks for all items.

        For each (pickup_coord, count) in `self.items`, creates `count` tasks
        pairing that pickup location with a randomly chosen drop-zone. Shuffles
        the resulting list so that task order is unpredictable.

        Returns: A randomized list of (pickup_coord, drop_coord)
        tuples, one per item instance in the warehouse.
        """
        
        # Build one task per item instance:
        tasks = [
            (pickup, self.random.choice(self.drop_coords))
            for pickup, count in self.items.items()
            for _ in range(count)
        ]
        
        # Shuffle tasks, so agents don't all head to the same zones in sequence:
        self.random.shuffle(tasks)
        return tasks

    def create_shelves(self, row_positions, shelf_edge_gap, aisle_interval):
        """
        Generate shelf locations in the warehouse grid, place Shelf agents there,
        and return the list of shelf coordinates.

        For each row in `row_positions`, shelves span from `shelf_edge_gap` columns
        in from the left edge to the same distance from the right edge, skipping
        two-column-wide aisles at every `aisle_interval`.

        Args:
        row_positions: Y-coordinates for each horizontal shelf row.
        shelf_edge_gap: Number of columns to leave empty at left/right edges.
        aisle_interval: Column spacing between aisles; defines two-column gaps
        at that interval.

        Returns: Coordinates (x, y) where Shelf agents were placed.
        """
        
        # Calculate the inclusive range for shelf placement:
        x0 = shelf_edge_gap
        x1 = self.width - shelf_edge_gap
        
        # Build list of all candidate shelf coordinates:
        coords = [
            (x, y)
            for y in row_positions
            for x in range(x0, x1)
            
            # Skip two-wide aisles:
            if not (aisle_interval and ((x - x0) % aisle_interval in (0, 1)))
        ]
        
        # Place a Shelf agent at each coordinate in the grid:
        for pos in coords:
            self.grid.place_agent(Shelf(self), pos)
        return coords
    
    def create_drop_zones(self, drop_coords):
        """
        Generate and deploy drop-zone cells in the warehouse grid.

        Determines “base” corner points, then expands each
        base into a square of (side length 2) * (self.drop_zone_size - 1),
        clipping to grid bounds. Places a DropZone agent at every cell in
        these squares and returns the full list of drop-zone coordinates.

        Args:
        drop_coords: Optional list of base (x, y) positions around which 
        to build drop-zones. If None, uses the four grid corners.

        Returns: All (x, y) coordinates where DropZone agents were placed.
        """
        # Determine base corner positions:
        bases = drop_coords or [
            (0, 0),
            (self.width - 1, 0),
            (0, self.height - 1),
            (self.width - 1, self.height - 1),
        ]
        
        # Compute expansion radius:
        r = self.drop_zone_size - 1

        # Collect all expanded coordinates - Within grid bounds:
        expanded = set()
        for cx, cy in bases:
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    x = min(max(cx + dx, 0), self.width - 1)
                    y = min(max(cy + dy, 0), self.height - 1)
                    expanded.add((x, y))

        # Place a drop-zone agent at each coordinate:
        for (x, y) in expanded:
            dz = DropZone(self)
            self.grid.place_agent(dz, (x, y))

        # return list of drop-zone coordinates:
        return list(expanded)

    def create_items(self, shelf_coords, default_count = 1):
        """
        Populate shelves with item agents and track their counts and instances.

        For each coordinate in `shelf_coords`, creates `default_count` new
        ShelfItem agents, places each one on the grid at that location, and
        records both:
        - `items`: how many items are at each shelf coordinate
        - `item_agents`: the actual ShelfItem instances at each coordinate

        Args:
        shelf_coords: Coordinates of shelf cells.
        default_count: Number of items to spawn per shelf cell.

        Returns:
            tuple:
                - First variable: Mapping from each shelf coordinate
                to the number of items placed there.
                - Second variable: Mapping from each shelf
                coordinate to the list of ShelfItem agent objects placed.
        """
        
        # Initialise item counts per shelf coordinate:
        items = {pos: default_count for pos in shelf_coords}
        
        # Prepare container for the agent instances:
        item_agents: dict[tuple[int,int], list] = {}
        
        # For each shelf position and it's count:
        # - Spawn 'count' ShelfItem agents at pos
        # - Record the list of agents at this position
        for pos, count in items.items():
            agents: list = []
            for _ in range(count):
                item = ShelfItem(self)
                self.grid.place_agent(item, pos)
                agents.append(item)
            item_agents[pos] = agents
        return items, item_agents

    def step(self):
        """
        Advance the simulation by one tick, performing all core per-step routines:

        1. Reset path cache
        2. Respawn shelf items if enabled and due
        3. Assign new tasks to idle agents
        4. Perform collision-avoidance updates
        5. Manually step each agent, catching and logging exceptions
        6. Update congestion metrics and evaporate pheromones
        7. Process each agent’s state machine (pickup→drop→relocate→idle)
        8. Collect tick-level data
        """
        
        # Record current simulation time:
        now = self.schedule.time

        # Reset per‐tick cache:
        self.path_cache.clear()

        # Handle respawning of shelf‐items:
        if self.respawn_enabled:
            while self.respawn_queue and self.respawn_queue[0][1] <= self.ticks:
                shelf_pos, scheduled_time = self.respawn_queue.popleft()
                # spawn new item agent
                new_item = ShelfItem(self)
                self.grid.place_agent(new_item, shelf_pos)
                # update counts
                self.items[shelf_pos] = self.items.get(shelf_pos, 0) + 1
                self.item_agents[shelf_pos].append(new_item)
                # add a new pickup→drop task
                self.tasks.append((shelf_pos, self.random.choice(self.drop_coords)))

        # Assign new tasks:
        self.apply_strategy()

        # Collision avoidance:
        self.update_agent_field()
        self.cleanup_reservations()
        self.handle_priority_yielding()

        # Advance all agents:
        moved = 0
        for agent in list(self.schedule.agents):
            if not isinstance(agent, WarehouseAgent):
                continue
            prepos = agent.pos
            try:
                agent.step()
            except Exception as e:
                print(f"[ERROR][{now}] Agent {agent.unique_id} EXCEPTION in step(): {e}")
            if agent.pos != prepos:
                moved += 1

        # Update metrics, evaporate pheromones:
        self.update_congestion_metrics()
        self.evaporate_pheromones()

        # Run the pickup→drop→relocate→idle state‐machine:
        for agent in self.schedule.agents:
            if isinstance(agent, WarehouseAgent):
                self.process_agent_state(agent)

        # Collect data & increment tick:
        self.collect_tick_data()

    def update_agent_field(self):
        """
        Compute a repulsive potential field based on current WarehouseAgent positions.

        Maintains `self.agent_field`, a dict mapping each grid cell (x, y) to a
        “repulsion” value. Each agent contributes:
        - +5 to its own cell
        - +2 to each of its 4-way neighbors

        The resulting field can be used by pathfinding or collision-avoidance
        routines to bias movement away from congested areas.
        """
        
        # Initialize field dictionary if it doesn't exist:
        if not hasattr(self, 'agent_field'):
            # Create only once with grid dimensions:
            self.agent_field = {(x, y): 0 for x in range(self.width) for y in range(self.height)}
        
        # Reset all cells repulsion to 0 before re-depositing:
        for cell in self.agent_field:
            self.agent_field[cell] = 0
            
        # For each agent, deposit repulsion:
        for agent in self.schedule.agents:
            if isinstance(agent, WarehouseAgent):
                x, y = agent.pos
                # Strong at agent position:
                self.agent_field[(x, y)] += 5
                # Weaker around:
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    local_neighbour = (x+dx, y+dy)
                    if local_neighbour in self.agent_field:
                        self.agent_field[local_neighbour] += 2

    def cleanup_reservations(self):
        """
        Purge expired time-space reservations, 
        retaining only those in the future.
        """
        
        # Get current simulation time from the scheduler:
        now = self.schedule.time

        # Rebuild the reservatons dictionary, keeping only future time slots:
        self.reservations = {k: v for k, v in self.reservations.items() if k[2] > now}
    
    def handle_priority_yielding(self):
        """
        Resolve next-step movement conflicts by priority yielding with array sorting.

        Scans all WarehouseAgent movement intents (their next path cell) and
        identifies collisions where multiple agents intend the same cell. For
        each collided group, attempts to insert an alternative sidestep into
        each agent’s path (preferring left, then any free neighbor). Finally,
        rebuilds time-space reservations for the next tick based on the
        potentially modified paths.
        """
        
        # Record current simulation time:
        now = self.schedule.time
        
        # Local assignment for additional clarity:
        import numpy as local_np

        # Collect each agent's intended next-cell index:
        agents = list(self.schedule.agents)
        n = len(agents)
        
        # Initialise all to -1 i.e. no intent:
        idxs = local_np.full(n, -1, dtype=local_np.int64)
        for i, a in enumerate(agents):
            if isinstance(a, WarehouseAgent) and a.path:
                x, y = a.path[0]
                idxs[i] = x * self.height + y

        # Sort indices to group identical intents together:
        order = local_np.argsort(idxs)
        sorted_idxs = idxs[order]

        # Scan for runs of duplicate non-negative indices i.e. collision groups:
        i = 0
        while i < n:
            j = i + 1
            # Find end of run of equal indices:
            while j < n and sorted_idxs[j] == sorted_idxs[i] and sorted_idxs[i] != -1:
                j += 1
            # If no more than one agent in run, then resolve collision:
            if j - i > 1:
                for k in order[i:j]:
                    a = agents[k]
                    x0, y0 = a.pos
                    # Attempt left side-step:
                    alt = (x0-1, y0)
                    if 0 <= alt[0] < self.width and self.grid.is_cell_empty(alt):
                        a.path.insert(0, alt)
                    else:
                        # Otherwise, try any other neighbouring free cell:
                        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
                            local_neighbour = (x0+dx, y0+dy)
                            if (0 <= local_neighbour[0] < self.width and 0 <= local_neighbour[1] < self.height
                                    and self.grid.is_cell_empty(local_neighbour)):
                                a.path.insert(0, local_neighbour)
                                break
            i = j

        # Rebuild reservations for time = now + 1, based on updated next steps:
        new_res = {}
        for a in agents:
            if isinstance(a, WarehouseAgent) and a.path:
                nx, ny = a.path[0]
                new_res[(nx, ny, now+1)] = a.unique_id
        self.reservations = new_res
            
    def update_congestion_metrics(self):
        """
        Update congestion statistics and heatmap based on agent occupancy.

        Scans the entire grid to find cells containing one or more WarehouseAgent
        instances. For each tick:
        1. Counts the number of congested cells and adds this to `self.congestion_accum`.
        2. Increments the heatmap count for each congested cell to track how often
            each location experiences congestion.
        """
        
        # Establish list to collect coordinates of occupied cells:
        congested_cells = []
        
        # Identify all cells with >= 1 WarehouseAgent:
        for contents, (x, y) in self.grid.coord_iter():
            # Filter cell contents for agent instances:
            robots = [a for a in contents if isinstance(a, WarehouseAgent)]
            if robots:
                congested_cells.append((x, y))
        
        # Add the count of congested cells to the running total:
        self.congestion_accum += len(congested_cells)

        # Increment heatmap counts for every congested cell:
        for cell in congested_cells:
            self.heatmap[cell] += 1
            
    def evaporate_pheromones(self):
        """
        Apply evaporation to the pheromone field across the entire grid.

        Reduces each cell’s pheromone concentration by the configured
        evaporation rate in one vectorized NumPy operation.
        """
        
        # Multiply all pheromone values by (1 - evaporation_rate):
        self.pheromone_field *= (1.0 - self.pheromone_evap_rate)        

    def apply_strategy(self):
        """
        Invoke the current task-assignment strategy method.

        Dynamically resolves and calls `<strategy>_strategy` based on
        `self.strategy`.
        """

        try:
            # Look up the method matching the strategy name:
            strategy_fn = getattr(self, f"{self.strategy}_strategy")
        except AttributeError:
            # The chosen strategy string does not correspond to any method:
            raise ValueError(f"Unknown strategy {self.strategy!r}")
        # Execute the resolved strategy function to assigned tasks:
        strategy_fn()

    def collect_tick_data(self):
        """
        Advance the simulation clock, collect performance data, and handle termination.

        1. Increments the internal tick counter.
        2. Records all model-level reporters via the DataCollector.
        3. If `self.max_steps` is reached or exceeded, exports data to 'results.csv'
        and flags the model as no longer running.
        """
        
        # Increment tick counter:
        self.ticks += 1
        
        # Collect into the DataCollector:
        self.datacollector.collect(self)
        
        # Auto-Export the dataset when the cap of max_steps has been hit:
        if self.max_steps is not None and self.ticks >= self.max_steps:
            # Write out all the model-level reporters to results.csv:
            self.datacollector.get_model_vars_dataframe().to_csv("results.csv", index=False)
            # Ensure that any external controller sees we're done:
            self.running = False
    
    def centralised_strategy(self):
        """
        Assign tasks to idle agents using a global cost-matrix and the Hungarian algorithm.

        This method:
        1. Gathers all idle WarehouseAgent instances and pending pickup→drop tasks.
        2. Builds vectorized NumPy arrays for agent positions, delivery counts, task pickup positions,
            neighbourhood adjacency, heatmap values, and pheromone values.
        3. Calls `make_cost_matrix` to compute a cost matrix balancing distance, load, heatmap, and pheromones.
        4. Solves the assignment problem via `linear_sum_assignment` for minimal total cost.
        5. For each valid (agent, task) pair, sets:
            - `agent.current_pickup` to the task’s pickup coordinate
            - `agent.pickup_pos` to the chosen neighbor entry
            - `agent.next_drop` to the task’s drop coordinate
            - `agent.path` via `compute_path` from current position to `pickup_pos`
            - `agent.state` to `"to_pickup"`
        6. Removes assigned tasks from `self.tasks`.
        """
        
        # Trade-off coefficients for heatmap (alpha) and pheromone (beta):
        alpha, beta = 0.1, 0.5

        # Gather all currently idle agents and pending tasks:
        idle = [a for a in self.schedule.agents
                if isinstance(a, WarehouseAgent)
                and getattr(a, "state", None) in (None, "idle")]
        if not idle or not self.tasks:
            return

        # Build NumPy arrays for agent data and task positions:
        import numpy as _np
        n_agents = len(idle)
        n_tasks  = len(self.tasks)

        # Agent positions and delivery counts:
        agent_pos = _np.empty((n_agents, 2), np.int64)
        deliveries = _np.empty(n_agents, np.float64)
        for i,a in enumerate(idle):
            agent_pos[i, 0], agent_pos[i, 1] = a.pos
            deliveries[i] = a.deliveries

        # Task pickup coordinates:
        task_pos = _np.empty((n_tasks, 2), np.int64)
        for j,(pickup,_) in enumerate(self.tasks):
            task_pos[j,0], task_pos[j,1] = pickup

        # Build fixed-size neighbour array for each task pickup:
        free_adj_lists = []
        maxL = 0
        for pickup,_ in self.tasks:
            lst = [pos for pos in self.neighbours[pickup] if self.is_passable(pos, pickup)]
            free_adj_lists.append(lst)
            if len(lst) > maxL:
                maxL = len(lst)

        neighbour_array = _np.full((n_tasks, maxL, 2), -1, np.int64)
        neighbour_lens= _np.zeros(n_tasks, np.int64)
        for j,lst in enumerate(free_adj_lists):
            neighbour_lens[j] = len(lst)
            for k,(x,y) in enumerate(lst):
                neighbour_array[j,k,0] = x
                neighbour_array[j,k,1] = y

        # Pack heatmap and pheromone fields into arrays:
        heatmap_vals   = _np.zeros((self.width, self.height), np.float64)
        pheromone_vals = _np.zeros((self.width, self.height), np.float64)
        for x in range(self.width):
            for y in range(self.height):
                heatmap_vals[x,y]   = self.heatmap[(x,y)]
                pheromone_vals[x,y] = self.pheromones[(x,y)]

        # Compute cost matrix and best neighbour indices via JIT helper:
        cost_mat, best_k = make_cost_matrix(
            agent_pos, task_pos,
            neighbour_array, neighbour_lens,
            heatmap_vals, pheromone_vals,
            deliveries, alpha, beta, self.gamma
        )

        # Solve the assignment problem for minimal total cost:
        rows, cols = linear_sum_assignment(cost_mat)
        to_remove = []
        for i,j in zip(rows, cols):
            if cost_mat[i,j] >= _np.inf:
                continue
            a    = idle[i]
            pickup,_ = self.tasks[j]
            # Get neighbour index, set it to k:
            k    = best_k[i,j]
            entry = (int(neighbour_array[j,k,0]), int(neighbour_array[j,k,1]))
            a.current_pickup = pickup
            a.pickup_pos     = entry
            a.next_drop      = self.tasks[j][1]
            a.path           = self.compute_path(a.pos, entry)
            a.state          = "to_pickup"
            to_remove.append(j)

        # Remove tasks that have just been assigned, in reverse index order:
        for idx in sorted(to_remove, reverse=True):
            self.tasks.pop(idx)

    def decentralised_strategy(self):
        """
        Assign tasks to idle agents via a decentralized bidding process based on proximity.

        1. Identify all idle WarehouseAgent instances and pending pickup→drop tasks.
        2. Compute Manhattan distances between each idle agent and each task’s pickup location.
        3. Form bids of 1/(distance+1) for tasks within `self.auction_radius`, zero otherwise.
        4. In descending bid order, assign each task to the highest-bidding available agent:
        - Select the passable neighbor of the pickup cell closest to the agent’s position.
        - Set agent.current_pickup, pickup_pos, next_drop, path (via compute_path), and state="to_pickup".
        5. Remove all assigned tasks from `self.tasks`.
        """
        
        # Gather idle agents and bail out if there's nothing to do:
        idle = [a for a in self.schedule.agents
                if isinstance(a, WarehouseAgent) and a.state in (None, 'idle')]
        if not idle or not self.tasks:
            return

        # Build NumPy array of agent positions (A) and task pickup coords (P):
        A = np.array([a.pos for a in idle], dtype=np.int64)
        P = np.array([pickup for pickup,_ in self.tasks], dtype=np.int64)

        # Compute Manhattan distance matrix D between each agent and each task:
        D = np.abs(A[:,None,0] - P[None,:,0]) + np.abs(A[:,None,1] - P[None,:,1])

        # Build bid matrix:
        bids = np.where(D <= self.auction_radius, 1.0/(D+1), 0.0)

        # Flatten bids in descending order to resolve highest bids first:
        flat_idxs = np.dstack(np.unravel_index(np.argsort(-bids.ravel()), bids.shape))[0]
        
        assigned_agents = set()
        assigned_tasks  = set()
        
        # Iterate through sorted bid pairs and assign non-conflicting wins:
        for ai, ti in flat_idxs:
            # Skip zero bids or agents/tasks already assigned:
            if bids[ai,ti] == 0 or ai in assigned_agents or ti in assigned_tasks:
                continue
            
            agent = idle[ai]
            pickup, drop_zone = self.tasks[ti]
            
            
            # Select the neighbour of pickup that is passable and nearest to agent:
            local_neighbours = [pos for pos in self.neighbours[pickup] 
                    if self.is_passable(pos, pickup)]
            entry = min(local_neighbours, key=lambda pos: abs(agent.pos[0]-pos[0]) + abs(agent.pos[1]-pos[1]))
            
            # Update agent routing and state:
            agent.current_pickup = pickup
            agent.pickup_pos     = entry
            agent.next_drop      = drop_zone
            agent.path           = self.compute_path(agent.pos, entry)
            agent.state          = 'to_pickup'
            assigned_agents.add(ai)
            assigned_tasks.add(ti)
            if len(assigned_agents) == len(idle):
                break

        # Remove assigned tasks from the task list:
        for idx in sorted(assigned_tasks, reverse=True):
            self.tasks.pop(idx)

    def swarm_strategy(self):
        """
        Coordinate idle robots in clusters to grab nearby tasks, with fallback exploration.

        1. Groups idle robots into spatial clusters via BFS within `pickup_radius`.
        2. For each cluster, matches members to tasks whose pickups lie within the cluster radius:
        - Removes assigned tasks, selects a passable neighbor cell closest to each robot,
            and updates robot.current_pickup, pickup_pos, next_drop, path, state, and reservations.
        3. If no cluster grabbed tasks, computes the swarm centroid and assigns the single best task
        to all idle robots for exploration.
        4. Deposits a small pheromone increment at every remaining pickup location.
        """
        
        # Record current simulation time:
        now = self.schedule.time
        
        # Set radius for cluster formation and task proximity:
        pickup_radius = 3

        # Cluster‐grab - Form clusters of idle robots within pickup_radius:
        idle_agents = [a for a in self.robots if a.state == "idle"]
        visited = set()
        for a in idle_agents:
            if a in visited:
                continue
            # BFS to collect all agents within pickup_radius of this seed:
            cluster = {a}
            queue = [a]
            while queue:
                u = queue.pop()
                for v in idle_agents:
                    if v not in cluster and (
                    abs(u.pos[0] - v.pos[0]) + abs(u.pos[1] - v.pos[1])
                    <= pickup_radius
                    ):
                        cluster.add(v)
                        queue.append(v)
            visited |= cluster

            # find tasks whose pickup coords lie within pickup_radius of any cluster member:
            candidates = [
                (idx, pickup, drop_zone)
                for idx, (pickup, drop_zone) in enumerate(self.tasks)
                if any(
                    abs(member.pos[0] - pickup[0]) + abs(member.pos[1] - pickup[1]) <= pickup_radius
                    for member in cluster
                )
            ]

            # Assign each candidate to one cluster member in arbitrary order:
            assignments = []
            for (idx, pickup, drop_zone), member in zip(candidates, cluster):
                assignments.append((idx, pickup, drop_zone, member))

            # Remove assigned tasks and configure each robot:
            for idx, pickup, drop_zone, member in sorted(assignments, key=lambda x: x[0], reverse=True):
                self.tasks.pop(idx)
                # Choose a passable neighbour of pickup closest to this member:
                local_neighbours = self.neighbours[pickup]
                free_adj = [pos for pos in local_neighbours if self.is_passable(pos, pickup)]
                if not free_adj:
                    continue
                entry = min(
                    free_adj,
                    key=lambda pos: abs(member.pos[0] - pos[0]) + abs(member.pos[1] - pos[1])
                )
                
                # Update agent routing and state:
                member.current_pickup = pickup
                member.pickup_pos = entry
                member.next_drop    = drop_zone 
                member.path         = self.compute_path(member.pos, entry)
                member.state        = "to_pickup"
                # Reserve the first step in time-space for next tick:
                if member.path:
                    self.reservations[(member.path[0][0], member.path[0][1], now+1)] = member.unique_id

        # Exploration fallback - If tasks remain, but no one is heading to pickup:
        if self.tasks and not any(a.state=="to_pickup" for a in self.robots):
            idle_agents = [a for a in self.robots if a.state=="idle"]
            if not idle_agents:
                # nothing left to explore
                return
            # Compute swarm centroid:
            cx = sum(a.pos[0] for a in idle_agents) / len(idle_agents)
            cy = sum(a.pos[1] for a in idle_agents) / len(idle_agents)
            # Select the single best task by centroid proximity:
            idx, (pickup, drop_zone ) = min(
                enumerate(self.tasks),
                key=lambda t: abs(cx - t[1][0][0]) + abs(cy - t[1][0][1])
            )
            # Determine neighbour entry for exploration:
            self.tasks.pop(idx)
            local_neighbours = self.neighbours[pickup]
            free_adj = [pos for pos in local_neighbours if self.is_passable(pos, pickup)]
            if free_adj:
                entry = min(
                    free_adj,
                    key=lambda pos: abs(cx - pos[0]) + abs(cy - pos[1])
                )
                # Assign this exploration task to all idle agents:
                for member in idle_agents:
                    member.current_pickup = pickup
                    member.pickup_pos    = entry
                    member.next_drop     = drop_zone
                    member.path          = self.compute_path(member.pos, entry)
                    member.state         = "to_pickup"
                    if member.path:
                        self.reservations[(member.path[0][0],
                                        member.path[0][1],
                                        now+1)] = member.unique_id

        # Deposit pickup pheromones at each remaining pickup location:
        for pickup, _ in self.tasks:
            self.pheromones[pickup] += 0.5
    
    def heuristic(self, a, b):
        """
        Estimate the cost between two grid cells using Manhattan distance.

        Args:
            a: The (x, y) coordinate of the first cell.
            b: The (x, y) coordinate of the second cell.

        Returns: The Manhattan distance |a.x - b.x| + |a.y - b.y|.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def is_passable(self, cell, goal):
        """
        Determine if a grid cell can be traversed, treating the goal as always passable.

        A cell is considered passable if:
        - It matches the `goal` coordinate, allowing agents to enter their target.
        - It contains no Shelf agents (static obstacles) and no WarehouseAgent instances
            (dynamic obstacles).

        Args:
            cell: The (x, y) coordinate of the cell to test.
            goal: The destination coordinate currently being targeted.

        Returns: True if the cell is free for movement or is the goal, False otherwise.
        """
        
        # Always allow movement into the goal cell, even if another agent is present i.e. ShelfAgent:
        if cell == goal:
            return True
        
        # Inspect all objects currently in the target cell:
        for obj in self.grid.get_cell_list_contents([cell]):
            # Block movement if there's a shelf, or another WarehouseAgent:
            if isinstance(obj, (Shelf, WarehouseAgent)):
                return False
        return True

    def compute_path(self, start, goal):
        """
        Compute a time-space reservation–aware A* path from `start` to `goal`.

        This search considers:
        - Static obstacles (Shelf agents)
        - Dynamic obstacles and future reservations
        - A time dimension to avoid stepping into reserved cells
        - A cap on expansions (`width*height*4`) to prevent infinite loops
        Results are cached per (start, goal) pair for reuse.

        Args:
            start: Starting grid coordinate.
            goal: Target grid coordinate.

        Returns: Ordered list of coordinates from just after `start` to `goal`.
        Returns [] if `start==goal`, no path found, or expansion limit hit.
        """
        
        # Record current simulation time:
        now = self.schedule.time

        # Quick exit if already at the goal:
        if start == goal:
            return []

        # Return cached path if the pair has been computed before:
        cache_key = (start, goal)
        if cache_key in self.path_cache:
            return list(self.path_cache[cache_key])

        # Initialize A* structures:
        start_time = now
        open_set = []
        
        # Push the start node with f = heuristic(start, goal):
        heapq.heappush(open_set, (self.heuristic(start, goal), (start[0], start[1], start_time)))
        came_from = {}
        # Holds the cost from start to each (x, y, t):
        g_score = { (start[0], start[1], start_time): 0 }

        # Prevent runaway expansions:
        expansions = 0
        max_expansions = self.width * self.height * 4

        # Main A* loop:
        while open_set:
            expansions += 1
            # Abort search if too many expansions occur:
            if expansions > max_expansions:
                return []

            # Pop the node with lowest f_score:
            _, (x, y, t) = heapq.heappop(open_set)
            
            # If it reaches the goal cell, record its time and then exit:
            if (x, y) == goal:
                goal_time = t
                break
            
            # Explore all time-space-allowed neighbours:
            for nx, ny in self.allowed_neighbours(x, y, t, goal):
                nt = t + 1
                tentative_g = g_score[(x, y, t)] + 1
                key = (nx, ny, nt)
                # If this path to neighbour is better, then record it:
                if tentative_g < g_score.get(key, inf):
                    came_from[key] = (x, y, t)
                    g_score[key] = tentative_g
                    f_score = tentative_g + self.heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f_score, key))
        else:
            # Open set exhausted without reaching goal:
            return []

        # Reconstruct the path by walking backwards from goal_time:
        path = []
        node = (goal[0], goal[1], goal_time)
        while (node[0], node[1], node[2]) != (start[0], start[1], start_time):
            path.append((node[0], node[1]))
            node = came_from[node]
        path.reverse()

        # Cache and return the computed path:
        self.path_cache[cache_key] = list(path)
        return path

    def random_empty_cell(self):
        """
        Choose a random grid cell that contains no shelves.

        Repeatedly samples coordinates until one is found whose contents
        do not include any Shelf agents, making it safe for spawning robots
        or items.

        Returns: A randomly selected (x, y) coordinate with no Shelf.
        """
        
        # Continuously sample until an empty (shelf-free) cell is found:
        while True:
            # Pick a random column and row within grid bounds:
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            
            # Inspect the cell's contents, and ensure no Shelf is present:
            if all(not isinstance(a, Shelf) for a in self.grid.get_cell_list_contents([(x,y)])):
                return x, y
            
    def allowed_neighbours(self, x, y, t, goal):
        """
        Determine valid 4-way neighbor cells reachable at time t+1.

        Considers an optional search_radius bounding box around the current
        position or the provided goal, filters out cells reserved by other
        agents at the next timestep, and excludes impassable obstacles
        (except the goal, which is always allowed).

        Args:
        x: Current x-coordinate.
        y: Current y-coordinate.
        t: Current time step.
        goal: Target cell to always treat as passable.
              If None, bounding box centers on (x, y).

        Returns: All neighbour coordinates satisfying:
                1) Within the search_radius bounding box (if set).
                2) Not reserved by another agent at time t+1.
                3) Passable according to `self.is_passable`, or equals `goal`.
        """
        
        # Compute bounding box limits based on search_radius and goal:
        if self.search_radius is not None:
            sr = self.search_radius
            # Determine centre for box: goal if provided, else current cell:
            x_min = max(min(x, (goal or (x,y))[0]) - sr, 0)
            x_max = min(max(x, (goal or (x,y))[0]) + sr, self.width - 1)
            y_min = max(min(y, (goal or (x,y))[1]) - sr, 0)
            y_max = min(max(y, (goal or (x,y))[1]) + sr, self.height - 1)
        else:
            # No bounding box - Allow full grid:
            x_min, x_max, y_min, y_max = 0, self.width-1, 0, self.height-1

        out = []
        # Iterate over the four-connected neighbours:
        for nx, ny in self.neighbours[(x, y)]:
            # Skip if outside the bounding box:
            if not (x_min <= nx <= x_max and y_min <= ny <= y_max):
                continue

            # Skip if another agent reserved this cell at t+1:
            res = self.reservations.get((nx, ny, t+1))
            if isinstance(res, int):
                continue

            # Skip if impassable, unless it's the goal:
            if not self.is_passable((nx, ny), goal):
                continue

            out.append((nx, ny))

        # Debug block:
        if not out:
            local_neighbours = self.neighbours[(x, y)]
            blocked = [(pos, self.reservations.get((pos[0], pos[1], t+1))) for pos in local_neighbours]
        return out

    def process_agent_state(self, agent: WarehouseAgent):
        """
        Advance a WarehouseAgent through its state transition workflow.

        The agent cycles through:
        1. to_pickup → idle if unreachable
        2. to_pickup → to_dropoff on successful pickup
        3. to_dropoff → relocating after dropoff (with optional respawn scheduling)
        4. relocating → idle once staging is complete

        Args:
        agent: The agent whose state and path are updated.
        """

        # Handle unreachable pickup - No path, but not yet at pickup coordinate:
        if agent.state == "to_pickup" and not agent.path and agent.pos != agent.pickup_pos:
            self.tasks.append((agent.current_pickup, agent.next_drop))
            agent.state = "idle"
            return

        # On arrival at pickup location:
        if agent.state == "to_pickup" and not agent.path:
            # Check if an item is available at this shelf:
            items_here = self.item_agents.get(agent.current_pickup, [])
            if not items_here:
                # No item to pick up. Instead, drop the task and go idle:
                agent.state = "idle"
                return
            # remove one item from the shelf and grid:
            itm = items_here.pop()
            self.grid.remove_agent(itm)
            self.items[agent.current_pickup] -= 1
            
            # Plan path from current position to the drop-zone:
            agent.path = self.compute_path_to_drop(agent.pos, agent.next_drop)
            agent.state = "to_dropoff"
            return

        # On arrival at drop-off location:
        if agent.state == "to_dropoff" and not agent.path:
            # Schedule respawn of the shelf item if enabled:
            if self.respawn_enabled:
                respawn_time = self.ticks + self.item_respawn_delay
                self.respawn_queue.append((agent.current_pickup, respawn_time))
            
            # Update delivery metrics:
            self.total_task_steps += agent.task_steps
            agent.task_steps = 0
            agent.deliveries += 1
            self.total_deliveries += 1

            # Choose a staging area and plot path there:
            staging = self.random_empty_cell()
            agent.path = self.compute_path(agent.pos, staging)
            agent.state = "relocating"
            return

        # After relocating to staging, return to idle state:
        if agent.state == "relocating" and not agent.path:
            agent.state = "idle"
            return
