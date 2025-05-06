# run_experiments.py (updated to ensure metrics + parameters in output CSV)
import os
import itertools
import multiprocessing
import pandas as pd
import random
from tqdm import tqdm  # progress bar
from model import WarehouseEnvModel

# --- Simulation runner for a single configuration ---
def single_run(params: dict) -> dict:
    """
    Run one simulation with given parameters headlessly.
    Returns a dict of model metrics combined with the input params.
    """
        # Extract and remove max_steps and iteration from params (defaults)
    max_steps = params.get("max_steps", 500)
    model_params = params.copy()
    model_params.pop("max_steps", None)
    model_params.pop("iteration", None)
    model = WarehouseEnvModel(**model_params)
    # Run for a fixed number of steps
    for _ in range(max_steps):
        model.step()
    # collect metrics directly from model
    result = {
        'total_deliveries':   model.total_deliveries,
        'ticks':              model.ticks,
        'pendingtasks':       len(model.tasks),
        'collisions':         model.collisions,
        'total_task_steps':   model.total_task_steps
    }
    # merge in input parameters (excluding internal max_steps)
    result.update(model_params)
    return result

# --- Batch runner to group multiple runs in one worker ---
def run_batch(params_batch: list[dict]) -> list[dict]:
    results = []
    for params in params_batch:
        try:
            result = single_run(params)
        except Exception as e:
            result = {**params, 'error': str(e)}
        results.append(result)
    return results

# --- Utility to split list into N roughly equal chunks ---
def chunk_list(lst: list, n_chunks: int) -> list[list]:
    chunk_size = (len(lst) + n_chunks - 1) // n_chunks
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

# --- Utility to write DataFrame chunks to Parquet ---
def write_batch(df_chunk: pd.DataFrame, path: str = "results.parquet"):
    df_chunk.to_parquet(path, index=False)

# Runner for a single permutation: executes multiple iterations
iterations = 15

def run_permutation(perm):
    """Execute multiple iterations for one parameter permutation."""
    results = []
    for it in range(iterations):
        params = {**perm, 'iteration': it}
        try:
            res = single_run(params)
        except Exception as e:
            res = {**params, 'error': str(e)}
        results.append(res)
    return results


# --- Main entry point ---
if __name__ == "__main__":
    # Define experimental parameter grid
    variable_params = {
        'strategy':       ['centralised', 'decentralised', 'swarm'],
        'num_agents':     [5, 10, 15, 20],
        'shelf_edge_gap': [1, 2, 3],
        'aisle_interval': [5, 10],
        'shelf_rows':     [3, 5, 7],
        'width':          [20, 30, 40],
        'height':         [20, 30, 40],
    }
    # Build list of unique permutations (no iteration)
    perms = [dict(zip(variable_params.keys(), vals))
             for vals in itertools.product(*variable_params.values())]
    # Stratified sampling
    max_samples = 200
    strategies = variable_params['strategy']
    samples_per_strat = max_samples // len(strategies)
    strat_samples = []
    for strat in strategies:
        strat_group = [p for p in perms if p['strategy'] == strat]
        if len(strat_group) <= samples_per_strat:
            strat_samples.extend(strat_group)
        else:
            strat_samples.extend(random.sample(strat_group, samples_per_strat))
    leftover = max_samples - len(strat_samples)
    if leftover > 0:
        remaining = [p for p in perms if p not in strat_samples]
        strat_samples.extend(random.sample(remaining, min(leftover, len(remaining))))
    original_count = len(perms)
    perms = strat_samples
    print(f"Sampling {len(perms)}/{original_count} unique permutations (≈{samples_per_strat} per strategy)")

    # Build full list of parameter sets including iterations
    all_params = []
    for perm in perms:
        for it in range(iterations):
            p = perm.copy()
            p['iteration'] = it
            all_params.append(p)

        # Parallelize over permutations directly, tracking each with tqdm
    n_workers = min(len(perms), multiprocessing.cpu_count())
    nested = []
    with multiprocessing.Pool(processes=n_workers) as pool:
        for sub in tqdm(pool.imap(run_permutation, perms), total=len(perms), desc="Permutations"):
            nested.append(sub)
    # Flatten results
    results = [r for sub in nested for r in sub]

    # Create DataFrame and save
    df = pd.DataFrame(results)
    if 'error' in df.columns:
        df = df.drop(columns=['error'])
    df.to_csv('batch_results.csv', index=False)
    print(f"Batch completed: {len(df)} rows written to batch_results.csv.")
