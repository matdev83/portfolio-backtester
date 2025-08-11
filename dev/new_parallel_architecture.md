# Project Plan: Refactoring the Parallel Optimization Architecture

This document outlines the plan to refactor the multiprocessing architecture in the portfolio backtester to improve performance and scalability.

## 1. The Problem: Data Serialization Bottleneck

The current parallel architecture, while functionally correct, suffers from a major performance and scalability bottleneck. It uses the `spawn` start method, which requires all arguments passed to a new worker process to be serialized (pickled).

The `ParallelOptimizationRunner` passes the entire `OptimizationData` object—containing large Pandas DataFrames of historical data—to each worker.

This leads to two critical issues:
1.  **High Serialization Overhead**: The main process spends significant CPU time and wall-clock time pickling the same large dataset for every worker, delaying the start of the actual optimization.
2.  **Massive Memory Duplication**: Each of the `N` worker processes holds a full, independent copy of the dataset in its RAM. This means for `N` workers, the memory usage is `N * <dataset_size>`, which severely limits scalability.

## 2. Possible Solutions Explored

1.  **Change Start Method to `fork()`**: Utilizes the operating system's Copy-on-Write (CoW) feature to share memory implicitly.
    -   *Con*: Not available on Windows, making it a non-starter for this project.

2.  **Use `multiprocessing.shared_memory`**: A low-level, cross-platform mechanism for true shared memory.
    -   *Con*: Prohibitively complex to implement for high-level objects like Pandas DataFrames, requiring manual deconstruction and reconstruction of underlying data buffers.

3.  **Worker-Side Data Initialization**: The parent process passes file paths, and each worker loads the data from disk.
    -   *Pro*: Simple and robust.
    -   *Con*: Can lead to redundant I/O, though this is often mitigated by the OS disk cache.

## 3. Chosen Solution: Worker-Side Init with Memory-Mapping

We will proceed with a hybrid of Solution 3 and your suggestion to use memory-mapping. This approach offers the best balance of performance, memory efficiency, and implementation complexity.

### Rationale

This solution directly addresses the core problems:
- **It eliminates the serialization bottleneck**: The parent process only pickles small strings (file paths) and metadata, which is instantaneous.
- **It eliminates the memory duplication problem**: By using `numpy.memmap`, all worker processes can access the same data file on disk. The operating system is smart enough to load the file's pages into physical RAM only once and share those pages across all processes. This gives us the primary benefit of true shared memory without the complexity.
- **It is cross-platform**: `numpy.memmap` is fully supported on Windows.

## 4. High-Level Architecture Plan

The data flow will be inverted from a "push" model to a "pull" model.

1.  **Parent Process (`ParallelOptimizationRunner`)**:
    -   Loads the complete `OptimizationData` object from the data source as it does now.
    -   **Deconstructs `OptimizationData`**:
        -   Extracts large, numerical `pd.DataFrame` objects (e.g., `daily`, `monthly`, `returns`).
        -   Converts the underlying numerical data of these DataFrames into NumPy arrays.
    -   **Saves Data to Temporary Files**:
        -   Saves the NumPy arrays to disk as memory-mappable `.npy` files.
        -   Saves the non-numerical metadata required to reconstruct the DataFrames (e.g., the `index` and `columns`) to a separate file using `pickle`.
    -   Creates a new small, serializable `OptimizationDataContext` object containing only the file paths and essential metadata.
    -   Starts the worker processes, passing only the lightweight `OptimizationDataContext` object.

2.  **Worker Process (`_optuna_worker`)**:
    -   Receives the `OptimizationDataContext` object.
    -   **Reconstructs `OptimizationData`**:
        -   Loads the pickled metadata (index, columns).
        -   Uses `numpy.memmap` to open the `.npy` data files **without loading them entirely into memory**.
        -   Re-assembles the Pandas DataFrames using the memory-mapped arrays and the loaded metadata.
    -   Proceeds with its assigned optimization trials using the reconstructed, memory-efficient data.
    -   The parent process is responsible for cleaning up the temporary files after all workers have finished.

## 5. Detailed Execution Plan

Here is a phased plan with actionable tasks to track our progress.

### Phase 1: Data Persistence and Context Creation
- [ ] Create a new dataclass `OptimizationDataContext` in `src/portfolio_backtester/optimization/results.py`. This class will contain file paths for the memory-mapped data and the pickled metadata.
- [ ] In `ParallelOptimizationRunner.run()`, before starting processes, implement the logic to deconstruct the `self.data` (`OptimizationData`) object.
- [ ] Save the numerical NumPy data from the core DataFrames (`daily`, `monthly`, `returns`) to temporary `.npy` files. Use the `tempfile` module to manage their creation.
- [ ] Save the corresponding `index` and `columns` metadata for each DataFrame to a temporary pickle file.
- [ ] Populate an instance of `OptimizationDataContext` with the paths to these new files.

### Phase 2: Worker Refactoring for Memory-Mapped Loading
- [ ] Modify the `_optuna_worker` function signature to accept the `OptimizationDataContext` object instead of the full `OptimizationData` object.
- [ ] Inside `_optuna_worker`, implement the data reconstruction logic.
- [ ] Load the pickled metadata (index, columns) from the file path provided in the context object.
- [ ] Use `numpy.memmap(path, mode='r')` to open the `.npy` data files.
- [ ] Re-create the `daily`, `monthly`, and `returns` Pandas DataFrames using the loaded metadata and the memory-mapped NumPy arrays.
- [ ] Assemble the reconstructed DataFrames back into a new `OptimizationData` object for use in the rest of the worker's logic.

### Phase 3: Integration and Verification
- [ ] Update the `Process` creation call inside `ParallelOptimizationRunner.run()` to pass the new `OptimizationDataContext` object to the worker.
- [ ] Add robust cleanup logic, likely in a `finally` block, within `ParallelOptimizationRunner.run()` to ensure the temporary data and metadata files are always deleted after the optimization completes or fails.
- [ ] Add logging to show when data is being saved and when workers are reconstructing it, to make the new process transparent.
- [ ] Run a multi-process optimization run (e.g., with `--n_jobs 4`) on the `dummy_signal_strategy` to confirm that the new architecture works correctly from end to end.

### Phase 4: Quality Assurance and Finalization
- [ ] Run `black`, `ruff`, and `mypy` on all modified files to ensure code quality and type safety.
- [ ] Review the changes for any potential edge cases (e.g., what happens if the input DataFrames are empty?).
- [ ] Add comments to the new logic explaining the memory-mapping strategy and its purpose.
