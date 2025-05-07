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
                    
    def compute_path_to_drop(self, start: tuple[int,int], goal: tuple[int,int]) -> list[tuple[int,int]]:
        """
        Greedy, reservation-aware walk from start → goal along the static_distance gradient.
        Falls back to full A* only if blocked.  Caps the loop to avoid infinite cycling.
        """
        now = self.schedule.time

        if goal not in self.drop_coords:
            return self.compute_path(start, goal)

        path = []
        x0, y0 = start
        t = now
        loop_count = 0
        max_loops = self.width * self.height * 4

        while (x0, y0) != goal:
            loop_count += 1
            if loop_count > max_loops:
                return []

            nbrs = self.neighbours[(x0, y0)]
            candidates = [
                (nx, ny) for nx, ny in nbrs
                if self.static_distance.get((nx, ny), inf) < self.static_distance[(x0, y0)]
                and not isinstance(self.reservations.get((nx, ny, t+1)), int)
                and self._is_passable((nx, ny), goal)
            ]
            if not candidates:
                return self.compute_path(start, goal)

            x1, y1 = min(candidates, key=lambda c: self.static_distance[c])
            path.append((x1, y1))
            # Use None placeholder reservation (now treated as non-blocking in A*)
            self.reservations[(x1, y1, t+1)] = None
            x0, y0, t = x1, y1, t + 1
        return path


    def spawn_robots(self, num_agents: int) -> list[WarehouseAgent]:
        """
        Create `num_agents` WarehouseAgent instances, add each to the
        scheduler, place it in a random empty cell, and return the list.
        """
        robots: list[WarehouseAgent] = []
        for _ in range(num_agents):
            robot = WarehouseAgent(self)
            self.schedule.add(robot)
            x, y = self.random_empty_cell()
            self.grid.place_agent(robot, (x, y))
            robots.append(robot)
        return robots

    def create_tasks(self) -> list[tuple[int,int]]:
        """
        Build one (pickup, drop) tuple per item, choosing
        a random drop-zone for each, then shuffle.
        """
        tasks = [
            (pickup, self.random.choice(self.drop_coords))
            for pickup, count in self.items.items()
            for _ in range(count)
        ]
        self.random.shuffle(tasks)
        return tasks

    def create_shelves(
        self,
        row_positions: list[int],
        shelf_edge_gap: int,
        aisle_interval: int
    ) -> list[tuple[int,int]]:
        """
        Build shelf coordinates with the given edge-gap and aisle-interval,
        place Shelf agents there, and return the coord list.
        """
        x0, x1 = shelf_edge_gap, self.width - shelf_edge_gap
        coords = [
            (x, y)
            for y in row_positions
            for x in range(x0, x1)
            
            # Skip two-wide aisles:
            if not (aisle_interval and ((x - x0) % aisle_interval in (0, 1)))
        ]
        for pos in coords:
            self.grid.place_agent(Shelf(self), pos)
        return coords
    
    def create_drop_zones(
        self,
        drop_coords: list[tuple[int,int]] | None = None
    ) -> list[tuple[int,int]]:
        """
        1) Determine the 4 “base” corners (or use `drop_coords` if provided).
        2) For each corner (cx,cy), build a (2s-1)x(2s-1) square around it,
           clipped to grid bounds, where s == self.drop_zone_size.
        3) Place a DropZone agent at every such cell.
        4) Return the full list of coords.
        """
        # 1) base corners
        bases = drop_coords or [
            (0, 0),
            (self.width - 1, 0),
            (0, self.height - 1),
            (self.width - 1, self.height - 1),
        ]
        # radius from center: size=1→r=0, size=2→r=1 (3×3), size=3→r=2 (5×5)
        r = self.drop_zone_size - 1

        # 2) build set of all expanded coords
        expanded = set()
        for cx, cy in bases:
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    x = min(max(cx + dx, 0), self.width - 1)
                    y = min(max(cy + dy, 0), self.height - 1)
                    expanded.add((x, y))

        # 3) place the agents
        for (x, y) in expanded:
            dz = DropZone(self)
            self.grid.place_agent(dz, (x, y))

        # 4) return as list
        return list(expanded)

    def create_items(
        self,
        shelf_coords: list[tuple[int,int]],
        default_count: int = 1
    ) -> tuple[dict[tuple[int,int],int], dict[tuple[int,int], list]]:
        """
        For each shelf cell:
          1) set its item‐count in self.items,
          2) place that many ShelfItem agents,
          3) record them in self.item_agents.
        Returns (items, item_agents).
        """
        items = {pos: default_count for pos in shelf_coords}
        item_agents: dict[tuple[int,int], list] = {}
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
        1) Clear path cache
        2) Handle respawns
        3) Assign new tasks
        4) Collision avoidance
        5) Advance agents (manually, with per-agent logging)
        6) Update metrics, pheromones, then state‐machine, data collection
        """
        now = self.schedule.time

        # 1️⃣ Reset per‐tick cache
        self.path_cache.clear()

        # 2️⃣ Handle respawning of shelf‐items
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

        # 3️⃣ Assign new tasks
        self.apply_strategy()

        # 4️⃣ Collision avoidance
        self._update_agent_field()
        self._cleanup_reservations()
        self._handle_priority_yielding()

        movable = 0
        blocked_desc = []
        for a in self.schedule.agents:
            if not isinstance(a, WarehouseAgent) or not a.path:
                continue
            nx, ny = a.path[0]
            reserver = self.reservations.get((nx, ny, now+1))
            if reserver is None or reserver == a.unique_id:
                movable += 1
            else:
                blocked_desc.append(f"Agent {a.unique_id} → {(nx,ny)} reserved by {reserver}")

        # 5️⃣ Advance all agents *manually* so we can see exactly who hangs
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

        # 6️⃣ Update metrics, evaporate pheromones
        self.update_congestion_metrics()
        self.evaporate_pheromones()

        # 7️⃣ Run the pickup→drop→relocate→idle state‐machine
        for agent in self.schedule.agents:
            if isinstance(agent, WarehouseAgent):
                self._process_agent_state(agent)

        # 8️⃣ Collect data & increment tick
        self.collect_tick_data()


    def _update_agent_field(self):
        """
        Build a repulsive potential field from current agents.
        """
        # Initialize field
        if not hasattr(self, 'agent_field'):
            # Create only once with grid dimensions
            self.agent_field = {(x, y): 0 for x in range(self.width) for y in range(self.height)}
        # Reset
        for cell in self.agent_field:
            self.agent_field[cell] = 0
        # Deposit repulsion
        for agent in self.schedule.agents:
            if isinstance(agent, WarehouseAgent):
                x, y = agent.pos
                # Strong at agent position
                self.agent_field[(x, y)] += 5
                # Weaker around
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nbr = (x+dx, y+dy)
                    if nbr in self.agent_field:
                        self.agent_field[nbr] += 2

    def _cleanup_reservations(self):
        """
        Remove outdated time-space reservations.
        """
        now = self.schedule.time

        # Keep only future slots
        self.reservations = {k: v for k, v in self.reservations.items() if k[2] > now}
    
    def _handle_priority_yielding(self):
        """
        Fast collision-avoidance via array sorting of next-cell intents.
        """
        now = self.schedule.time
        import numpy as _np

        # 1) Build array of size n_agents with desired cell idx or -1
        agents = list(self.schedule.agents)
        n = len(agents)
        idxs = _np.full(n, -1, dtype=_np.int64)
        for i, a in enumerate(agents):
            if isinstance(a, WarehouseAgent) and a.path:
                x, y = a.path[0]
                idxs[i] = x * self.height + y

        # 2) Sort to group duplicates
        order = _np.argsort(idxs)
        sorted_idxs = idxs[order]

        # 3) Detect runs of the same idx >1 => collisions
        i = 0
        while i < n:
            j = i + 1
            while j < n and sorted_idxs[j] == sorted_idxs[i] and sorted_idxs[i] != -1:
                j += 1
            if j - i > 1:
                # collision among agents[order[i:j]]
                for k in order[i:j]:
                    a = agents[k]
                    x0, y0 = a.pos
                    # try left
                    alt = (x0-1, y0)
                    if 0 <= alt[0] < self.width and self.grid.is_cell_empty(alt):
                        a.path.insert(0, alt)
                    else:
                        # fallback an adjacent sidestep
                        for dx, dy in [(0,1),(1,0),(0,-1),(-1,0)]:
                            nb = (x0+dx, y0+dy)
                            if (0 <= nb[0] < self.width and 0 <= nb[1] < self.height
                                    and self.grid.is_cell_empty(nb)):
                                a.path.insert(0, nb)
                                break
            i = j

        # 4) Rebuild reservations for t+1
        new_res = {}
        for a in agents:
            if isinstance(a, WarehouseAgent) and a.path:
                nx, ny = a.path[0]
                new_res[(nx, ny, now+1)] = a.unique_id
        self.reservations = new_res
            
    def update_congestion_metrics(self):
        """
        Update model’s congestion statistics and heatmap.
        1. Computes number of cells occupied by at least one agent.
        2. Adds that count to self.congestion_accum.
        3. Increments self.heatmap at each occupied cell.
        """
        congested_cells = []
        for contents, (x, y) in self.grid.coord_iter():
            robots = [a for a in contents if isinstance(a, WarehouseAgent)]
            if robots:
                congested_cells.append((x, y))
        
        self.congestion_accum += len(congested_cells)

        # **new**: increment heatmap counts for every cell that had ≥1 robot
        for cell in congested_cells:
            self.heatmap[cell] += 1
            
    def evaporate_pheromones(self):
        """
        Evaporate the pheromone field in one NumPy operation.
        """
        # C-level evaporation across entire grid
        self.pheromone_field *= (1.0 - self.pheromone_evap_rate)        

    def apply_strategy(self):
        """
        Dispatch to the correct strategy method, or error if unknown.
        """
        try:
            strategy_fn = getattr(self, f"{self.strategy}_strategy")
        except AttributeError:
            raise ValueError(f"Unknown strategy {self.strategy!r}")
        strategy_fn()

    def collect_tick_data(self):
        """
        Increment tick count and collect the model’s data.
        """
        
        # Increment your tick counter:
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
        Assign tasks via a bidding process based on inverse Manhattan distance.

        1. Each idle agent bids for pickups within auction_radius.
        2. Highest bids win; winning agents get assigned pickup_pos, next_drop, path and state.
        3. Won tasks are removed from self.tasks.
        """
        alpha, beta = 0.1, 0.5

        # 1️⃣ gather idle agents
        idle = [a for a in self.schedule.agents
                if isinstance(a, WarehouseAgent)
                and getattr(a, "state", None) in (None, "idle")]
        if not idle or not self.tasks:
            return

        # 2️⃣ convert to numpy
        import numpy as _np
        n_agents = len(idle)
        n_tasks  = len(self.tasks)

        agent_pos = _np.empty((n_agents, 2), np.int64)
        deliveries = _np.empty(n_agents, np.float64)
        for i,a in enumerate(idle):
            agent_pos[i, 0], agent_pos[i, 1] = a.pos
            deliveries[i] = a.deliveries

        task_pos = _np.empty((n_tasks, 2), np.int64)
        for j,(pu,_) in enumerate(self.tasks):
            task_pos[j,0], task_pos[j,1] = pu

        # 3️⃣ build fixed‐size neighbor array
        free_adj_lists = []
        maxL = 0
        for pu,_ in self.tasks:
            lst = [pos for pos in self.neighbours[pu] if self._is_passable(pos, pu)]
            free_adj_lists.append(lst)
            if len(lst) > maxL:
                maxL = len(lst)

        neigh_arr = _np.full((n_tasks, maxL, 2), -1, np.int64)
        neigh_lens= _np.zeros(n_tasks, np.int64)
        for j,lst in enumerate(free_adj_lists):
            neigh_lens[j] = len(lst)
            for k,(x,y) in enumerate(lst):
                neigh_arr[j,k,0] = x
                neigh_arr[j,k,1] = y

        # 4️⃣ pack heatmap & pheromone
        heatmap_vals   = _np.zeros((self.width, self.height), np.float64)
        pheromone_vals = _np.zeros((self.width, self.height), np.float64)
        for x in range(self.width):
            for y in range(self.height):
                heatmap_vals[x,y]   = self.heatmap[(x,y)]
                pheromone_vals[x,y] = self.pheromones[(x,y)]

        # 5️⃣ call JIT matrix builder
        cost_mat, best_k = make_cost_matrix(
            agent_pos, task_pos,
            neigh_arr, neigh_lens,
            heatmap_vals, pheromone_vals,
            deliveries, alpha, beta, self.gamma
        )

        # 6️⃣ solve assignment
        rows, cols = linear_sum_assignment(cost_mat)
        to_remove = []
        for i,j in zip(rows, cols):
            if cost_mat[i,j] >= _np.inf:
                continue
            a    = idle[i]
            pu,_ = self.tasks[j]
            # get the neighbor index k
            k    = best_k[i,j]
            entry = (int(neigh_arr[j,k,0]), int(neigh_arr[j,k,1]))
            a.current_pickup = pu
            a.pickup_pos     = entry
            a.next_drop      = self.tasks[j][1]
            a.path           = self.compute_path(a.pos, entry)
            a.state          = "to_pickup"
            to_remove.append(j)

        # 7️⃣ purge tasks
        for idx in sorted(to_remove, reverse=True):
            self.tasks.pop(idx)

    
    def decentralised_strategy(self):
        now = self.schedule.time
        idle = [a for a in self.schedule.agents
                if isinstance(a, WarehouseAgent) and a.state in (None, 'idle')]
        if not idle or not self.tasks:
            return

        # 1️⃣ build arrays
        A = np.array([a.pos for a in idle], dtype=np.int64)     # (nA,2)
        P = np.array([pu   for pu,_ in self.tasks], dtype=np.int64)  # (nT,2)

        # 2️⃣ compute distances
        D = np.abs(A[:,None,0] - P[None,:,0]) + np.abs(A[:,None,1] - P[None,:,1])  # (nA,nT)

        # 3️⃣ build bid matrix
        bids = np.where(D <= self.auction_radius, 1.0/(D+1), 0.0)

        # 4️⃣ pick winners in descending bid order
        # flatten with (agent_idx, task_idx) pairs
        flat_idxs = np.dstack(np.unravel_index(np.argsort(-bids.ravel()), bids.shape))[0]
        assigned_agents = set()
        assigned_tasks  = set()
        for ai, ti in flat_idxs:
            if bids[ai,ti] == 0 or ai in assigned_agents or ti in assigned_tasks:
                continue
            agent = idle[ai]
            pu, dr = self.tasks[ti]
            # choose best free_adj entry as before (or reuse precomputed neighbor arrays)
            neighs = [pos for pos in self.neighbours[pu] 
                    if self._is_passable(pos, pu)]
            entry = min(neighs, key=lambda pos: abs(agent.pos[0]-pos[0]) + abs(agent.pos[1]-pos[1]))
            agent.current_pickup = pu
            agent.pickup_pos     = entry
            agent.next_drop      = dr
            agent.path           = self.compute_path(agent.pos, entry)
            agent.state          = 'to_pickup'
            assigned_agents.add(ai)
            assigned_tasks.add(ti)
            if len(assigned_agents) == len(idle):
                break

        # 5️⃣ remove assigned tasks (descending index)
        for idx in sorted(assigned_tasks, reverse=True):
            self.tasks.pop(idx)


    def swarm_strategy(self):
        """
        Cluster‐based “collective grab” for idle robots within radius,
        then fallback centroid‐explore if none picked,
        then leave pickup/drop to the state‐machine.
        """
        now = self.schedule.time
        pickup_radius = 3

        # Cluster‐grab
        idle_agents = [a for a in self.robots if a.state == "idle"]
        visited = set()
        for a in idle_agents:
            if a in visited:
                continue
            # BFS to form cluster
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

            # find candidate tasks within radius
            candidates = [
                (idx, pu, dr)
                for idx, (pu, dr) in enumerate(self.tasks)
                if any(
                    abs(member.pos[0] - pu[0]) + abs(member.pos[1] - pu[1]) <= pickup_radius
                    for member in cluster
                )
            ]

            # pair them up (in arbitrary cluster order) but collect first
            assignments = []
            for (idx, pu, dr), member in zip(candidates, cluster):
                assignments.append((idx, pu, dr, member))

            # now remove tasks in descending idx order
            for idx, pu, dr, member in sorted(assignments, key=lambda x: x[0], reverse=True):
                self.tasks.pop(idx)
                neighs = self.neighbours[pu]
                free_adj = [pos for pos in neighs if self._is_passable(pos, pu)]
                if not free_adj:
                    continue
                entry = min(
                    free_adj,
                    key=lambda pos: abs(member.pos[0] - pos[0]) + abs(member.pos[1] - pos[1])
                )
                member.current_pickup = pu
                member.pickup_pos = entry
                member.next_drop    = dr
                member.path         = self.compute_path(member.pos, entry)
                member.state        = "to_pickup"
                if member.path:
                    self.reservations[(member.path[0][0], member.path[0][1], now+1)] = member.unique_id

        # Exploration fallback if nobody grabbed yet
        if self.tasks and not any(a.state=="to_pickup" for a in self.robots):
            idle_agents = [a for a in self.robots if a.state=="idle"]
            if not idle_agents:
                # nothing left to explore
                return
            cx = sum(a.pos[0] for a in idle_agents) / len(idle_agents)
            cy = sum(a.pos[1] for a in idle_agents) / len(idle_agents)
            # pick the single best task for the whole swarm
            idx, (pu, dr) = min(
                enumerate(self.tasks),
                key=lambda t: abs(cx - t[1][0][0]) + abs(cy - t[1][0][1])
            )
            # remove it once
            self.tasks.pop(idx)
            neighs = self.neighbours[pu]
            free_adj = [pos for pos in neighs if self._is_passable(pos, pu)]
            if free_adj:
                entry = min(
                    free_adj,
                    key=lambda pos: abs(cx - pos[0]) + abs(cy - pos[1])
                )
                for member in idle_agents:
                    member.current_pickup = pu
                    member.pickup_pos    = entry
                    member.next_drop     = dr
                    member.path          = self.compute_path(member.pos, entry)
                    member.state         = "to_pickup"
                    if member.path:
                        self.reservations[(member.path[0][0],
                                        member.path[0][1],
                                        now+1)] = member.unique_id

        # Deposit pickup pheromones
        for pu, _ in self.tasks:
            self.pheromones[pu] += 0.5

    
    def _heuristic(self, a: tuple[int,int], b: tuple[int,int]) -> int:
        """Manhattan distance on a 4-way grid."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _is_passable(self, cell: tuple[int,int], goal: tuple[int,int]) -> bool:
        """
        Returns True if `cell` is free to move into.
        Blocks shelves and other robots, except if `cell == goal`.
        """
        if cell == goal:
            return True
        for obj in self.grid.get_cell_list_contents([cell]):
            if isinstance(obj, (Shelf, WarehouseAgent)):
                return False
        return True

    def compute_path(self, start: tuple[int,int], goal: tuple[int,int]) -> list[tuple[int,int]]:
        """
        Time-space reservation-aware A* search from start to goal.
        Avoids shelves, robots, and reserved future cells.
        Aborts after a fixed maximum number of expansions.
        """
        now = self.schedule.time

        if start == goal:
            return []

        cache_key = (start, goal)
        if cache_key in self.path_cache:
            return list(self.path_cache[cache_key])

        # Initialize
        start_time = now
        open_set = []
        heapq.heappush(open_set, (self._heuristic(start, goal), (start[0], start[1], start_time)))
        came_from = {}
        g_score = { (start[0], start[1], start_time): 0 }

        expansions = 0
        max_expansions = self.width * self.height * 4

        while open_set:
            expansions += 1
            if expansions > max_expansions:
                return []

            _, (x, y, t) = heapq.heappop(open_set)
            if (x, y) == goal:
                goal_time = t
                break

            for nx, ny in self._allowed_neighbors(x, y, t, goal):
                nt = t + 1
                tentative_g = g_score[(x, y, t)] + 1
                key = (nx, ny, nt)
                if tentative_g < g_score.get(key, inf):
                    came_from[key] = (x, y, t)
                    g_score[key] = tentative_g
                    f_score = tentative_g + self._heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f_score, key))
        else:
            return []

        # Reconstruct path
        path = []
        node = (goal[0], goal[1], goal_time)
        while (node[0], node[1], node[2]) != (start[0], start[1], start_time):
            path.append((node[0], node[1]))
            node = came_from[node]
        path.reverse()

        self.path_cache[cache_key] = list(path)
        return path


    def random_empty_cell(self):
        """
        Return a random grid coordinate not occupied by a Shelf.

        Returns: (x, y) of an empty cell suitable for spawning or staging.
        """
        while True:
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            # Only return cells that have no Shelf in them
            if all(not isinstance(a, Shelf) for a in self.grid.get_cell_list_contents([(x,y)])):
                return x, y
            
    def _allowed_neighbors(self, x: int, y: int, t: int, goal: tuple[int,int] | None = None) -> list[tuple[int,int]]:
        """
        Return all 4-way neighbours of (x,y) at time t+1 that
        1) lie within the optional search_radius bounding box,
        2) aren’t reserved by *another* agent at t+1,
        3) are passable (unless they’re the specified goal).
        """
        # 1) bounding‐box:
        if self.search_radius is not None:
            sr = self.search_radius
            x_min = max(min(x, (goal or (x,y))[0]) - sr, 0)
            x_max = min(max(x, (goal or (x,y))[0]) + sr, self.width - 1)
            y_min = max(min(y, (goal or (x,y))[1]) - sr, 0)
            y_max = min(max(y, (goal or (x,y))[1]) + sr, self.height - 1)
        else:
            x_min, x_max, y_min, y_max = 0, self.width-1, 0, self.height-1

        out = []
        for nx, ny in self.neighbours[(x, y)]:
            # 1) within box?
            if not (x_min <= nx <= x_max and y_min <= ny <= y_max):
                continue

            # 2) block only if reserved by another agent (an int)
            res = self.reservations.get((nx, ny, t+1))
            if isinstance(res, int):
                continue

            # 3) passable (or it’s the goal cell)
            if not self._is_passable((nx, ny), goal):
                continue

            out.append((nx, ny))

        # debug if completely blocked
        if not out:
            nbrs = self.neighbours[(x, y)]
            blocked = [(pos, self.reservations.get((pos[0], pos[1], t+1))) for pos in nbrs]
        return out


    def _process_agent_state(self, agent: WarehouseAgent):
        """
        Advance agent through the pickup → dropoff → relocating → idle logic,
        with logging of every transition.
        """
        now = self.schedule.time
        prev_state = agent.state

        # 1️⃣ unreachable en‐route → return task
        if agent.state == "to_pickup" and not agent.path and agent.pos != agent.pickup_pos:
            self.tasks.append((agent.current_pickup, agent.next_drop))
            agent.state = "idle"
            return

        # 2️⃣ arrived at shelf → pick + plan drop
        if agent.state == "to_pickup" and not agent.path:
            # make sure there actually is an item waiting here
            items_here = self.item_agents.get(agent.current_pickup, [])
            if not items_here:
                # nothing to pick up — drop this task and go idle
                agent.state = "idle"
                return
            itm = items_here.pop()
            self.grid.remove_agent(itm)
            self.items[agent.current_pickup] -= 1
            agent.path = self.compute_path_to_drop(agent.pos, agent.next_drop)
            agent.state = "to_dropoff"
            return

        # 3️⃣ arrived at drop → record + staging + schedule respawn
        if agent.state == "to_dropoff" and not agent.path:
            if self.respawn_enabled:
                # enqueue this shelf cell for a delayed respawn
                respawn_time = self.ticks + self.item_respawn_delay
                self.respawn_queue.append((agent.current_pickup, respawn_time))

            self.total_task_steps += agent.task_steps
            agent.task_steps = 0
            agent.deliveries += 1
            self.total_deliveries += 1

            staging = self.random_empty_cell()
            agent.path = self.compute_path(agent.pos, staging)
            agent.state = "relocating"
            return

        # 4️⃣ finished staging → go idle
        if agent.state == "relocating" and not agent.path:
            agent.state = "idle"
            return
