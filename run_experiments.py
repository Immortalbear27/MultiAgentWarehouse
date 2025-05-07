import os
import itertools
import multiprocessing
import pandas as pd
import random
from tqdm import tqdm  # progress bar
from model import WarehouseEnvModel
from copy import deepcopy

# Cache of “static” model skeletons to speed up repeated __init__
_MODEL_CACHE = {}

# --- Simulation runner for a single configuration ---
def single_run(params: dict) -> dict:
    """
    Run one simulation with given parameters headlessly.
    Returns a dict of model metrics combined with the input params.
    """
    max_steps = params.get("max_steps", 500)
    iteration = params.get("iteration", 0)

    # Extract model construction params (exclude control fields)
    model_params = params.copy()
    model_params.pop("max_steps", None)
    model_params.pop("iteration", None)

    # Build static skeleton key and cache if needed
    static_params = model_params.copy()
    key = tuple(sorted(static_params.items()))
    if key not in _MODEL_CACHE:
        # Instantiate base model once per unique config
        _MODEL_CACHE[key] = WarehouseEnvModel(**static_params)
    model = deepcopy(_MODEL_CACHE[key])

    # OVERWRITE RNG with a fresh instance for each iteration
    new_rng = random.Random(iteration)
    # Assign to Mesa's RNG attribute
    model.random = new_rng
    # If an internal RNG used elsewhere
    if hasattr(model, '_rng'):
        model._rng = new_rng

    # Regenerate any RNG-dependent initial state (e.g. tasks ordering)
    try:
        model.tasks = model.create_tasks()
    except AttributeError:
        pass

    # Run simulation
    for step in range(max_steps):
        model.step()
        if not model.tasks:
            model.ticks = step + 1
            break
    else:
        model.ticks = max_steps

    # Collect metrics
    result = {
        'total_deliveries':   model.total_deliveries,
        'ticks':              model.ticks,
        'pendingtasks':       len(model.tasks),
        'collisions':         model.collisions,
        'total_task_steps':   model.total_task_steps
    }
    # Merge in input parameters for traceability
    result.update(model_params)
    result['iteration'] = iteration
    return result

# --- Batch runner to group multiple runs in one worker ---
def run_batch(params_batch: list[dict]) -> list[dict]:
    results = []
    for params in params_batch:
        try:
            results.append(single_run(params))
        except Exception as e:
            results.append({**params, 'error': str(e)})
    return results

# --- Utility to split list into N roughly equal chunks ---
def chunk_list(lst: list, n_chunks: int) -> list[list]:
    chunk_size = (len(lst) + n_chunks - 1) // n_chunks
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

# --- Main entry point ---
if __name__ == "__main__":
    # Define parameter grid
    variable_params = {
        'strategy':       ['centralised', 'decentralised', 'swarm'],
        'num_agents':     [5, 10, 15, 20],
        'shelf_edge_gap': [1, 2, 3],
        'aisle_interval': [5, 10],
        'shelf_rows':     [3, 5, 7],
        'width':          [20, 30, 40],
        'height':         [20, 30, 40],
    }
    # Generate unique permutations
    perms = [dict(zip(variable_params.keys(), vals))
             for vals in itertools.product(*variable_params.values())]

    # Stratified sampling cap
    max_samples = 200
    strategies = variable_params['strategy']
    samples_per_strat = max_samples // len(strategies)
    strat_samples = []
    for strat in strategies:
        group = [p for p in perms if p['strategy'] == strat]
        strat_samples.extend(group if len(group) <= samples_per_strat else random.sample(group, samples_per_strat))
    leftover = max_samples - len(strat_samples)
    if leftover > 0:
        remaining = [p for p in perms if p not in strat_samples]
        strat_samples.extend(random.sample(remaining, min(leftover, len(remaining))))
    original_count = len(perms)
    perms = strat_samples
    print(f"Sampling {len(perms)}/{original_count} unique permutations (≈{samples_per_strat} per strategy)")

    # Expand with iterations
    iterations = 30
    all_params = []
    for perm in perms:
        for it in range(iterations):
            p = perm.copy()
            p['iteration'] = it
            all_params.append(p)

    # Parallel execution setup
    n_workers = min(len(all_params), multiprocessing.cpu_count())
    batches = chunk_list(all_params, n_workers)
    print(f"Running {len(all_params)} runs across {len(batches)} batches on {n_workers} workers")

    # Execute and collect
    results = []
    with multiprocessing.Pool(processes=n_workers) as pool:
        for batch in tqdm(pool.imap(run_batch, batches), total=len(batches), desc="Batches"):
            results.extend(batch)

    # Save to CSV
    df = pd.DataFrame(results)
    if 'error' in df:
        df = df.drop(columns=['error'])
    df.to_csv('batch_results.csv', index=False)
    print(f"Batch completed: {len(df)} rows written to batch_results.csv.")
