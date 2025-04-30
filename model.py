# model.py

from mesa import Model
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from agent import Shelf, DropZone, WarehouseAgent, ShelfItem
from collections import deque
import heapq
from math import inf
from scipy.optimize import linear_sum_assignment



class WarehouseEnvModel(Model):
    """
    Warehouse with static shelves/drop‐zones + one moving agent.
    """
    def __init__(
        self,
        width = 30,
        height = 30,
        shelf_coords = None,
        drop_coords = None,
        drop_zone_size = 1,
        shelf_rows = 5,
        shelf_edge_gap = 1,
        aisle_interval = 5,
        num_agents = 3,
        strategy = "centralised",
        auction_radius = 10,
        seed = None
    ):
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.strategy = strategy
        self.num_agents = num_agents
        self.drop_zone_size = drop_zone_size
        self.auction_radius = auction_radius
        self.shelf_edge_gap = shelf_edge_gap
        self.aisle_interval = aisle_interval
        
        # Initialise the grid and scheduler:
        self.grid     = MultiGrid(width, height, torus=False)
        self.schedule = SimultaneousActivation(self)
        
        # keep a running count of how often each cell has ≥1 robot
        self.heatmap = {
            (x, y): 0
            for x in range(self.width)
            for y in range(self.height)
        }
        
        # Pheromone parameters
        self.pheromones = {
            (x, y): 0.0
            for x in range(self.width)
            for y in range(self.height)
        }
        self.pheromone_deposit  = 1.0   # how much each robot leaves per step
        self.pheromone_evap_rate = 0.05 # fraction to evaporate each tick
        self.gamma               = 0.5  # weight for pheromone attraction in cost

        
        # 1️⃣ Counters for metrics
        self.total_deliveries = 0        # Amount of total deliveries made
        self.collisions = 0              # Counter for collisions
        self.congestion_accum = 0        # Cumulative congestion counter
        self.ticks = 0                   # Used for averaging
        self.total_task_steps = 0        # Counter for total steps taken for tasks
        

        # DataCollector reporters:
        self.datacollector = DataCollector(
            model_reporters={
                "TotalDeliveries":      lambda m: m.total_deliveries,
                "AvgStepsPerDelivery":(
                    lambda m: m.total_task_steps / m.total_deliveries
                    if m.total_deliveries > 0 else 0
                ),
                "Collisions":      lambda m: m.collisions,
                "PendingTasks":    lambda m: len(m.tasks),
            }
        )

        # Default static layout
        # Compute the Y-positions of each shelf row
        row_positions = [
            int(round((i+1) * height / (shelf_rows+1)))
            for i in range(shelf_rows)
        ]
        
        # Initialise the shelf agents:
        self.shelf_coords = self.create_shelves(row_positions, self.shelf_edge_gap, self.aisle_interval)
        
        # Initialise the drop zone agents:
        self.drop_coords = self.create_drop_zones(drop_coords)
            
        # 5️⃣ Initialize items: 1 item per shelf cell:        
        self.items, self.item_agents = self.create_items(self.shelf_coords)

        # 6️⃣ Build initial task list: (pickup_pos, random_drop_pos)
        #    One task per item, with drop randomly chosen
        self.tasks = self.create_tasks()

        # Shuffle so tasks come in random order
        self.random.shuffle(self.tasks)

        # 1️⃣ Spawn multiple robots
        self.robots = self.spawn_robots(self.num_agents)
        
        # Create per-robot task bags:
        self.swarm_tasks: dict[WarehouseAgent, list[tuple, tuple]] = {
            robot: [] for robot in self.robots
        }
        for idx, task in enumerate(self.tasks):
            # Round-robin assignment:
            robot = self.robots[idx % len(self.robots)]
            self.swarm_tasks[robot].append(task)

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
        1) Collision-avoidance pre-processing
        2) Advance all agents one step.
        3) Update congestion & collision metrics.
        4) Assign new tasks according to strategy.
        5) Evaporate pheromones & collect end-of-tick data.
        """
        
        # Assignment of tasks and selection of strategy:
        self.apply_strategy()
        
        # 1️⃣ Collision avoidance: build repulsive field, clean reservations, handle yields
        self._update_agent_field()
        self._cleanup_reservations()
        self._handle_priority_yielding()

        # 2️⃣ Movement & task logic
        self.schedule.step()

        # 3️⃣ Metrics & post-step updates
        self.update_congestion_metrics()
        self.evaporate_pheromones()
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
        if not hasattr(self, 'reservations'):
            self.reservations = {}
        # Keep only future slots
        self.reservations = {k: v for k, v in self.reservations.items() if k[2] > now}

    def _handle_priority_yielding(self):
        """
        Detect conflicts on next move and apply local sidesteps & reserve future cells.
        """
        """
        Simple forced-left collision avoidance:
        1️⃣ If two or more agents intend the same cell, whoever can move left does so.
        2️⃣ Others proceed or sidestep if left is blocked.
        3️⃣ Reserve next cells.
        """
        now = self.schedule.time
        # Build next move intents
        next_moves = {}
        for agent in self.schedule.agents:
            if isinstance(agent, WarehouseAgent) and agent.path:
                dest = agent.path[0]
                next_moves.setdefault(dest, []).append(agent)

        for dest, agents in next_moves.items():
            if len(agents) > 1:
                # Force left preference
                for agent in agents:
                    # attempt move left relative to current heading
                    # assume left is x-1
                    left_cell = (agent.pos[0] - 1, agent.pos[1])
                    if 0 <= left_cell[0] < self.width and 0 <= left_cell[1] < self.height:
                        if self.grid.is_cell_empty(left_cell):
                            # insert left move before intended move
                            agent.path.insert(0, left_cell)
                            # replan remaining path to original goal
                            agent.path = self.compute_path(agent.pos, agent.path[-1])
                            continue
                # Placeholder collision metric update here:
                
                # If left moves not possible, fallback to random sidestep
                for agent in agents:
                    dirs = [(0,1),(1,0),(0,-1),(-1,0)]
                    self.random.shuffle(dirs)
                    for dx, dy in dirs:
                        alt = (agent.pos[0]+dx, agent.pos[1]+dy)
                        if 0 <= alt[0] < self.width and 0 <= alt[1] < self.height and self.grid.is_cell_empty(alt):
                            agent.path.insert(0, alt)
                            agent.path = self.compute_path(agent.pos, agent.path[-1])
                            break
        # Reserve next cells to prevent head-on collisions
        if not hasattr(self, 'reservations'):
            self.reservations = {}
        for agent in self.schedule.agents:
            if isinstance(agent, WarehouseAgent) and agent.path:
                dest = agent.path[0]
                # Reserve immediate next cell at t+1
                self.reservations[(dest[0], dest[1], now+1)] = agent.unique_id

        # --- New: Reserve second-step cell at t+2 to avoid pass-through ---
        if len(agent.path) > 1:
            second = agent.path[1]
            self.reservations[(second[0], second[1], now+2)] = agent.unique_id
    def update_congestion_metrics(self):
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
        rho = self.pheromone_evap_rate
        for cell, lvl in self.pheromones.items():
            self.pheromones[cell] = lvl * (1 - rho)

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
        self.ticks += 1
        self.datacollector.collect(self)

        
    def centralised_strategy(self):
        """
        1) Optimal assignment of idle agents → pickup‐adjacent cell (with congestion penalty).
        2) Phase 2: when agent arrives at pickup, remove item & plan drop leg.
        3) Phase 3: when agent arrives at drop, record delivery & vacate.
        4) Reactive re‐assignment: if a pickup becomes unreachable mid‐task, return it.
        """
        
        # Hyper-parameters:
        alpha = 0.1  # heat‐penalty weight
        beta = 0.5   # how strongly to “punish” already‐busy agents

        # ── Phase 1: optimal match for idle agents ──────────────────
        idle = [
            a for a in self.schedule.agents
            if isinstance(a, WarehouseAgent)
            and getattr(a, "state", None) in (None, "idle")
        ]
        if idle and self.tasks:
            cost_matrix = []
            path_cands  = []
            for agent in idle:
                costs = []
                cands = []
                for pickup, drop in self.tasks:
                    neighs = self.grid.get_neighborhood(
                        pickup, moore=False, include_center=False
                    )
                    free_adj = [
                        pos for pos in neighs
                        if not any(isinstance(x, Shelf)
                                for x in self.grid.get_cell_list_contents([pos]))
                    ]
                    best_entries = []
                    for n in free_adj:
                        path = self.compute_path(agent.pos, n)
                        dist = len(path)
                        heat_cost = sum(self.heatmap.get(cell, 0) for cell in path)
                        fairness_cost = beta * agent.deliveries
                        pher_cost = sum(self.pheromones[cell] for cell in path)
                        total_cost = dist + alpha * heat_cost + fairness_cost - self.gamma * pher_cost
                        best_entries.append((total_cost, n, path))
                    if best_entries:
                        cost, pos_sel, path_sel = min(best_entries, key=lambda x: x[0])
                    else:
                        cost, pos_sel, path_sel = float("inf"), None, []
                    costs.append(cost)
                    cands.append((pos_sel, path_sel))
                cost_matrix.append(costs)
                path_cands.append(cands)

            rows, cols = linear_sum_assignment(cost_matrix)
            to_remove = []
            for r, c in zip(rows, cols):
                if cost_matrix[r][c] == float("inf"):
                    continue
                agent      = idle[r]
                pickup, drop     = self.tasks[c]
                pickup_pos, path = path_cands[r][c]
                agent.current_pickup = pickup
                agent.pickup_pos     = pickup_pos
                agent.next_drop      = drop
                agent.path           = path
                agent.state          = "to_pickup"
                to_remove.append(c)
            for idx in sorted(set(to_remove), reverse=True):
                self.tasks.pop(idx)

        # ── Phase 2 / 3 / 4: pickup, drop, vacate & reactive re-assign ───────────
        for agent in [a for a in self.schedule.agents if isinstance(a, WarehouseAgent)]:
            # ❶ Reactive: if en-route to pickup but path empty *and* not at pickup → unreachable
            if agent.state == "to_pickup" and not agent.path and agent.pos != agent.pickup_pos:
                # give task back to the pool
                self.tasks.append((agent.current_pickup, agent.next_drop))
                agent.state = "idle"
                continue

            # ❷ Arrived at shelf → pick & plan dropoff
            if agent.state == "to_pickup" and not agent.path:
                item = self.item_agents[agent.current_pickup].pop()
                self.grid.remove_agent(item)
                self.items[agent.current_pickup] -= 1
                agent.path  = self.compute_path(agent.pos, agent.next_drop)
                agent.state = "to_dropoff"

            # ❸ Arrived at drop zone → record & plan vacate
            elif agent.state == "to_dropoff" and not agent.path:
                self.total_task_steps  += agent.task_steps
                agent.task_steps        = 0
                agent.deliveries += 1
                self.total_deliveries  += 1
                staging = self.random_empty_cell()
                agent.path  = self.compute_path(agent.pos, staging)
                agent.state = "relocating"

            # ❹ Finished vacating → truly idle
            elif agent.state == "relocating" and not agent.path:
                agent.state = "idle"
    
    def decentralised_strategy(self):
        """
        Auction-based decentralised task allocation:
        1) Idle robots within auction_radius bid for nearby tasks.
        2) Task goes to highest bidder (highest bid = 1/(distance+1)).
        3) Phase 2/3: pick up & drop as before.
        """

        # 1️⃣ Collect idle agents and unassigned tasks
        idle = [
            a for a in self.schedule.agents
            if isinstance(a, WarehouseAgent)
            and getattr(a, "state", None) in (None, "idle")
        ]
        if idle and self.tasks:
            # --- build & sort auction bids ---
            bids = []
            for t_idx, (pickup, drop) in enumerate(self.tasks):
                for agent in idle:
                    dist = abs(agent.pos[0] - pickup[0]) + abs(agent.pos[1] - pickup[1])
                    if dist <= self.auction_radius:
                        bids.append((1.0/(dist+1), agent, t_idx))
            bids.sort(key=lambda x: -x[0])

            # --- assign winners ---
            assigned = {}
            assigned_agents = set()
            for _, agent, t_idx in bids:
                if (agent not in assigned_agents
                        and agent.state in (None, "idle")
                        and t_idx not in assigned):
                    assigned[t_idx] = agent
                    assigned_agents.add(agent)
                    if len(assigned_agents) == len(idle):
                        break

            # --- if nobody bid, fall back to nearest-task for all idle robots ---
            if not assigned:
                for agent in idle:
                    # find the closest remaining task
                    best_idx, (best_pu, best_dr) = min(
                        enumerate(self.tasks),
                        key=lambda it: (
                            abs(agent.pos[0] - it[1][0][0]) + 
                            abs(agent.pos[1] - it[1][0][1])
                        )
                    )
                    # assign immediately
                    assigned[best_idx] = agent

            # --- commit whichever assignments we have ---
            to_remove = []
            for t_idx, agent in assigned.items():
                pickup, drop = self.tasks[t_idx]
                # existing commit logic follows unchanged
                neighs = self.grid.get_neighborhood(pickup, moore=False, include_center=False)
                free_adj = [
                    pos for pos in neighs
                    if not any(isinstance(x, Shelf) for x in self.grid.get_cell_list_contents([pos]))
                ]
                best_adj = min(free_adj, key=lambda pos: abs(pos[0] - agent.pos[0]) + abs(pos[1] - agent.pos[1]))
                agent.current_pickup = pickup
                agent.pickup_pos = best_adj
                agent.next_drop = drop
                agent.path = self.compute_path(agent.pos, best_adj)
                agent.state = "to_pickup"
                to_remove.append((pickup, drop))

            # remove assigned tasks from the pool
            for task in to_remove:
                self.tasks.remove(task)

        # Phase 2 & 3: handle pickups and dropoffs as before
        for agent in [a for a in self.schedule.agents if isinstance(a, WarehouseAgent)]:
            if agent.state == "to_pickup" and not agent.path:
                item = self.item_agents[agent.current_pickup].pop()
                self.grid.remove_agent(item)
                self.items[agent.current_pickup] -= 1
                agent.path = self.compute_path(agent.pos, agent.next_drop)
                agent.state = "to_dropoff"

            elif agent.state == "to_dropoff" and not agent.path:
                self.total_task_steps += agent.task_steps
                agent.task_steps = 0
                agent.state = "idle"
                self.total_deliveries += 1


    
    def swarm_strategy(self):
        """
        Each robot looks only at its own bag of tasks, in order.
        When it’s idle, it pops the next (pickup, drop) pair and executes
        the same 3‐phase logic as centralised.
        """
        for agent in self.robots:
            # Phase 1: pick up new task if idle
            if getattr(agent, "state", None) in (None, "idle") and self.swarm_tasks[agent]:
                pickup, drop = self.swarm_tasks[agent].pop(0)

                # find a free adjacent cell next to the shelf
                neighbors = self.grid.get_neighborhood(
                    pickup, moore=False, include_center=False
                )
                free_adj = [
                    pos for pos in neighbors
                    if not any(isinstance(x, Shelf)
                            for x in self.grid.get_cell_list_contents([pos]))
                ]
                # pick the neighbor closest (manhattan) to the agent
                best_adj = min(
                    free_adj,
                    key=lambda pos: abs(agent.pos[0] - pos[0]) +
                                    abs(agent.pos[1] - pos[1])
                )

                agent.current_pickup = pickup
                agent.pickup_pos     = best_adj
                agent.next_drop      = drop
                agent.path           = self.compute_path(agent.pos, best_adj)
                agent.state          = "to_pickup"

            # Phase 2: arrived next to shelf
            elif agent.state == "to_pickup" and not agent.path:
                item = self.item_agents[agent.current_pickup].pop()
                self.grid.remove_agent(item)
                self.items[agent.current_pickup] -= 1

                agent.path  = self.compute_path(agent.pos, agent.next_drop)
                agent.state = "to_dropoff"

            # Phase 3: arrived at drop
            elif agent.state == "to_dropoff" and not agent.path:
                self.total_task_steps   += agent.task_steps
                agent.task_steps         = 0
                agent.state             = "idle"
                self.total_deliveries   += 1
    
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

    def compute_path(
        self,
        start: tuple[int,int],
        goal:  tuple[int,int]
    ) -> list[tuple[int,int]]:
        """
        Time-space reservation-aware A* search from start to goal.
        Avoids shelves, robots, and reserved future cells.
        Returns list of coords excluding start.
        """
        if start == goal:
            return []

        # Starting time-layer
        start_time = self.schedule.time
        # open_set holds tuples (f_score, (x, y, t))
        open_set = []
        heapq.heappush(open_set, (self._heuristic(start, goal), (start[0], start[1], start_time)))
        came_from: dict[tuple[int,int,int], tuple[int,int,int]] = {}
        g_score = {(start[0], start[1], start_time): 0}

        while open_set:
            _, (x, y, t) = heapq.heappop(open_set)
            if (x, y) == goal:
                goal_time = t
                break

            for nbr in self.grid.get_neighborhood((x, y), moore=False, include_center=False):
                nx, ny = nbr
                nt = t + 1
                # Skip reserved cells at time nt
                if hasattr(self, 'reservations') and (nx, ny, nt) in self.reservations:
                    # reservation-aware: cannot enter this cell at this time
                    continue
                # Skip impassable by original logic
                if not self._is_passable((nx, ny), goal):
                    continue
                tentative_g = g_score[(x, y, t)] + 1
                key = (nx, ny, nt)
                if tentative_g < g_score.get(key, inf):
                    came_from[key] = (x, y, t)
                    g_score[key] = tentative_g
                    f_score = tentative_g + self._heuristic((nx, ny), goal)
                    heapq.heappush(open_set, (f_score, key))
        else:
            # No path found
            return []

        # Reconstruct path from goal_time back to start
        path: list[tuple[int,int]] = []
        node = (goal[0], goal[1], goal_time)
        while (node[0], node[1], node[2]) != (start[0], start[1], start_time):
            path.append((node[0], node[1]))
            node = came_from[node]
        path.reverse()
        return path

    def random_empty_cell(self):
        while True:
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            # Only return cells that have no Shelf in them
            if all(not isinstance(a, Shelf) for a in self.grid.get_cell_list_contents([(x,y)])):
                return x, y
