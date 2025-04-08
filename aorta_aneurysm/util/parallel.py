"""
SPDX-FileCopyrightText: Copyright 2024 German Cancer Research Center (DKFZ) and Institute of Radiology, Uniklinikum Erlangen, Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU).
SPDX-License-Identifier: CC BY-NC 4.0
"""

import os, multiprocessing
from typing import List, Tuple, Any

from tqdm.contrib.concurrent import process_map

def _worker_wrapper(args):
    worker = args[0]
    real_args = args[1:]
    return worker(*real_args)

def run_in_parallel(
        worker: callable,
        work: List[Tuple[Any]],
        num_workers: int = 8,
        verbose: bool = False,
    ) -> List[Any]:
    """
    Run code in parallel.

    Parameters
    ----------
    worker: callable
        The processing function, e.g. 'def worker(a,b): return a+b'
    work: List[Tuple[Any]]
        A list of arguments for each execution of worker, e.g. [(1,2), (3,4), (5,6)]
    num_workers : int
        How many workers to use
    verbose : bool
        Whether to display a progress bar

    Returns
    -------
    List[Any]
        A list of results of each execution (use 'return' at each call of worker, otherwise a list of None)
        E.g. with the sample worker above result would be [3,7,11]
    """
    if os.getenv("DEBUG", False) or os.getenv("DEBUG_SHOW_ALL_THREAD_ARGS", False): #DEBUG
        print(f"[INFO] Running in parallel in main thread because of DEBUG mode...")
        results = []
        for idx, x in enumerate(work):
            if idx==0:
                print(f"[INFO] \t\tSample first thread args: {x}")
            elif os.getenv("DEBUG_SHOW_ALL_THREAD_ARGS", False):
                print(f"[INFO] \t\tThread args: {x}")
            results.append(worker(*x))
        return results

    if not verbose:
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.starmap(worker, work)
            pool.close()
            pool.join()
        return results
    else: # verbose
        print(f"[INFO] Running in parallel with {num_workers} workers...")
        wrapped_work = [(worker, *x) for x in work]
        results = process_map(_worker_wrapper, wrapped_work, max_workers=num_workers, chunksize=(1 if len(wrapped_work)<=1000 else 10))
        return results
