# model.py

from mesa import Model
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from agent import Shelf, DropZone, WarehouseAgent, ShelfItem
import heapq
from collections import deque
from agent import Shelf

class WarehouseEnvModel(Model):
    """
    Warehouse with static shelves/drop‐zones + one moving agent.
    """
    def __init__(
        self,
        width: int = 30,
        height: int = 30,
        shelf_coords: list[tuple[int,int]] = None,
        drop_coords:  list[tuple[int,int]] = None,
        shelf_density: float = 0.3,
        shelf_rows: int = 5,
        shelf_edge_gap: int = 1,
        aisle_interval: int = 5,
        num_agents: int = 3,
        seed:         int = None
    ):
        super().__init__(seed=seed)
        self.grid     = MultiGrid(width, height, torus=False)
        self.schedule = SimultaneousActivation(self)
        
        # 1️⃣ Counters for metrics
        self.total_deliveries = 0        # throughput
        self.total_steps = 0             # for energy proxy
        self.collisions = 0              # collision count
        self.congestion_accum = 0        # cumulative congestion
        self.ticks = 0                   # for averaging

        # 2️⃣ DataCollector reporters
        self.datacollector = DataCollector(
            model_reporters={
                "Throughput":      lambda m: m.total_deliveries,
                "AvgEnergy":       lambda m: (m.total_steps / max(1, len(m.schedule.agents))),
                "Collisions":      lambda m: m.collisions,
                "AvgCongestion":   lambda m: (m.congestion_accum / max(1, m.ticks)),
                "PendingTasks":    lambda m: len(m.tasks),
            }
        )

        # Default static layout
        # Compute the Y-positions of each shelf row
        row_positions = [
            int(round((i+1) * height / (shelf_rows+1)))
            for i in range(shelf_rows)
        ]

        # Build shelf coordinates with edge gaps and optional internal aisles
        shelf_coords = []
        for y in row_positions:
            for x in range(shelf_edge_gap, width - shelf_edge_gap):
                # if aisle_interval is set, skip every Nth column
                if aisle_interval and ((x - shelf_edge_gap) % aisle_interval == 0):
                    continue
                shelf_coords.append((x, y))
        
        if drop_coords is None:
            drop_coords = [(0, 0), (width - 1, height - 1)]

        # Place static agents
        for pos in shelf_coords:
            s = Shelf(self)
            self.grid.place_agent(s, pos)
            
        # 5️⃣ Initialize items: 1 item per shelf cell
        #    You could later vary this per-shelf or have multiple items.
        self.items = { pos: 1 for pos in shelf_coords }
        self.item_agents = {}  # map pos → list of item‐agents

        for pos, count in self.items.items():
            self.item_agents[pos] = []
            for _ in range(count):
                item = ShelfItem(self)
                self.grid.place_agent(item, pos)
                self.item_agents[pos].append(item)

        # 6️⃣ Build initial task list: (pickup_pos, random_drop_pos)
        #    One task per item, with drop randomly chosen
        self.tasks = []
        for pickup in shelf_coords:
            for _ in range(self.items[pickup]):
                drop = self.random.choice(drop_coords)
                self.tasks.append((pickup, drop))

        # Shuffle so tasks come in random order
        self.random.shuffle(self.tasks)

        for pos in drop_coords:
            dz = DropZone(self)
            self.grid.place_agent(dz, pos)

        # 1️⃣ Spawn multiple robots
        for _ in range(num_agents):
            robot = WarehouseAgent(self)
            self.schedule.add(robot)
            x, y = self.random_empty_cell()
            self.grid.place_agent(robot, (x, y))

    def step(self):
        """
        1) Move every agent one step along its current path.
        2) After they've moved, assign new pickup or drop‐off paths.
        """
        
        # 1) Reset per‐tick stats
        cell_counts = {}  # to measure congestion (#agents per cell)
        for cell_contents, (x, y) in self.grid.coord_iter():
            count = len(cell_contents)
            if count > 1:
                cell_counts[(x,y)] = count
        
        # ── 1) Advance agents ─────────────────────────
        self.schedule.step()
        
         # 3) Count collisions: any cell where >1 agent landed
        #    (excluding Shelf/DropZone so only multiple robots count)
        for (x,y), count in cell_counts.items():
            # if two or more WarehouseAgent ended in same cell
            agents_here = [a for a in self.grid.get_cell_list_contents([(x,y)])
                           if isinstance(a, WarehouseAgent)]
            if len(agents_here) > 1:
                self.collisions += 1

        # 4) Measure congestion this tick
        self.congestion_accum += len(cell_counts)

        # ── 2) Centralised assignment ─────────────────
        for agent in [a for a in self.schedule.agents if isinstance(a, WarehouseAgent)]:
            # Phase 1: just finished moving and was 'idle' at start of tick?
            if getattr(agent, "state", None) in (None, "idle") and self.tasks:
                # assign only the pickup leg
                pickup, drop = self.tasks.pop(0)
                agent.current_pickup = pickup
                agent.next_drop = drop
                agent.path = self.compute_path(agent.pos, pickup)
                agent.state = "to_pickup"
                # no movement this tick (they already moved above)
        
            # Phase 2: arrived at pickup because path is now empty
            elif agent.state == "to_pickup" and not agent.path:
                # remove the visual item
                item = self.item_agents[agent.current_pickup].pop()
                self.grid.remove_agent(item)
                self.items[agent.current_pickup] -= 1

                # assign only the drop leg
                agent.path = self.compute_path(agent.pos, agent.next_drop)
                agent.state = "to_dropoff"

            # Phase 3: arrived at drop-off
            elif agent.state == "to_dropoff" and not agent.path:
                agent.state = "idle"
                self.total_deliveries += 1
                
        # 7) Finally, collect the data for this tick
        self.ticks += 1
        self.datacollector.collect(self)
    
    def compute_path(self, start, goal):
        """
        BFS on the 4-way grid, allowing movement *onto* the goal even if it
        has a Shelf agent. All other Shelf‐cells remain blocked.
        """
        if start == goal:
            return []

        queue = deque([start])
        came_from = {start: None}

        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for nbr in self.grid.get_neighborhood(current, moore=False, include_center=False):
                if nbr in came_from:
                    continue
                # **Only** skip shelf‐cells if *not* our destination
                contents = self.grid.get_cell_list_contents([nbr])
                if nbr != goal and any(isinstance(a, Shelf) for a in contents):
                    continue
                came_from[nbr] = current
                queue.append(nbr)

        # If we never reached the goal, return empty path
        if goal not in came_from:
            return []

        # Reconstruct the path (excluding start)
        path = []
        node = goal
        while node != start:
            path.append(node)
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
