# run_experiments.py
"""
Optimized batch experiment runner using chunked multiprocessing
and streaming results to Parquet without external dependencies.
"""
import os
import itertools
import multiprocessing
import pandas as pd
import glob
from model import WarehouseEnvModel

# --- Simulation runner for a single configuration ---
def single_run(params: dict) -> dict:
    """
    Run one simulation with given parameters headlessly.
    Returns a dict of model metrics combined with the input params.
    """
    model = WarehouseEnvModel(**params, max_steps=params.get("max_steps", 500))
    while model.running:
        model.step()
    df = model.datacollector.get_model_vars_dataframe()
    last = df.iloc[-1].to_dict() if not df.empty else {}
    last.update(params)
    return last

# --- Batch runner to group multiple runs in one worker ---
def run_batch(params_batch: list[dict]) -> list[dict]:
    """
    Run a list of parameter dicts sequentially in one process.
    """
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
    """
    Divide lst into n_chunks sublists of (approximately) equal size.
    """
    chunk_size = (len(lst) + n_chunks - 1) // n_chunks
    chunks = []
    for i in range(0, len(lst), chunk_size):
        chunks.append(lst[i : i + chunk_size])
    return chunks

# --- Utility to write DataFrame chunks to Parquet ---
def write_batch(df_chunk: pd.DataFrame, path: str = "results.parquet"):
    """
    Append or write a DataFrame to a Parquet file efficiently.
    """
    df_chunk.to_parquet(path, index=False)

# --- Utility to convert Parquet to CSV ---
def convert_parquet_to_csv(parquet_path: str = "results.parquet", csv_path: str = "batch_results.csv"):
    """
    Read a Parquet file and write its contents to a CSV file.
    Overwrites csv_path if it exists.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file '{parquet_path}' does not exist.")
    df = pd.read_parquet(parquet_path)
    df.to_csv(csv_path, index=False)
    print(f"Converted '{parquet_path}' to '{csv_path}', {len(df)} rows written.")

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

    iterations = 30
    all_configs = [
        {**dict(zip(variable_params.keys(), vals)), **{'iteration': it}}
        for it in range(iterations)
        for vals in itertools.product(*variable_params.values())
    ]

    n_workers = multiprocessing.cpu_count()
    batches = chunk_list(all_configs, n_workers)

    results_path = 'results.parquet'
    if os.path.exists(results_path):
        os.remove(results_path)

    # Directory for batch parquet files
    out_dir = 'batch_parquets'
    os.makedirs(out_dir, exist_ok=True)

    # Remove any existing parquet files
    for f in os.listdir(out_dir):
        if f.endswith('.parquet'):
            os.remove(os.path.join(out_dir, f))

    # Run workers and write each batch to its own Parquet via pandas
    with multiprocessing.Pool(processes=n_workers) as pool:
        for idx, batch_results in enumerate(pool.imap_unordered(run_batch, batches), start=1):
            # Create DataFrame from batch results
            df_batch = pd.DataFrame(batch_results)
            # Append to combined results CSV
            write_batch(df_batch, path=results_path)
            # Also save this batch as a separate Parquet file for later
            batch_dir = 'batch_parquets'
            os.makedirs(batch_dir, exist_ok=True)
            batch_file = os.path.join(batch_dir, f'batch_results_{idx}.parquet')
            df_batch.to_parquet(batch_file, index=False)
            print(f"Finished batch {idx}/{len(batches)}: {len(df_batch)} rows written; saved to {batch_file}")

    print(f"All {len(batches)} batches complete. Combined results saved to {results_path}.")
    # Combine batch parquet files into a single CSV
    csv_path = 'batch_results.csv'
    parquet_files = sorted(glob.glob(os.path.join(out_dir, 'batch_results_*.parquet')))
    dfs = [pd.read_parquet(f) for f in parquet_files]
    combined = pd.concat(dfs, ignore_index=True)
    # Drop 'error' column if present (from iteration kwarg issues)
    if 'error' in combined.columns:
        combined = combined.drop(columns=['error'])
    combined.to_csv(csv_path, index=False)
    print(f"Combined CSV written to {csv_path} with {len(combined)} rows.")