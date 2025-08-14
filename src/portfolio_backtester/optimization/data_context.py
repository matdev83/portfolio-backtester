"""
Optimization data context management for memory-efficient worker processes.

This module provides utilities to create and manage memory-mapped data contexts
for parallel optimization, reducing serialization overhead and memory usage.
"""

import os
import pickle
import tempfile
from typing import Dict, Union, Optional
import numpy as np
import pandas as pd
from loguru import logger

from .results import OptimizationData, OptimizationDataContext


class DataContextManager:
    """Manages memory-mapped data contexts for parallel optimization.

    This class handles the creation, storage, and cleanup of memory-mapped
    data files to reduce serialization overhead when passing large datasets
    to worker processes.
    """

    def __init__(self):
        self._temp_dir = None

    def create_context(self, data: OptimizationData) -> OptimizationDataContext:
        """Creates a memory-mapped context from OptimizationData.

        Args:
            data: The full optimization data to convert to memory-mapped format.

        Returns:
            A lightweight context object with paths to memory-mapped files.
        """
        # Create a temporary directory to store the memory-mapped files
        self._temp_dir = tempfile.TemporaryDirectory(prefix="ga_optimizer_")

        data_map = {
            "daily": data.daily,
            "monthly": data.monthly,
            "returns": data.returns,
        }

        metadata: Dict[str, Optional[Dict[str, Union[pd.Index, pd.MultiIndex]]]] = {}
        paths = {}

        for name, df in data_map.items():
            if df.empty:
                logger.debug(f"DataFrame '{name}' is empty, skipping.")
                paths[f"{name}_data_path"] = ""
                metadata[name] = None  # Explicitly mark as not having metadata
                continue

            # Save numerical data as a memory-mappable .npy file
            data_path = os.path.join(self._temp_dir.name, f"{name}_data.npy")
            np.save(data_path, df.to_numpy())
            paths[f"{name}_data_path"] = data_path

            # Store metadata (index, columns) for reconstruction
            metadata[name] = {"index": df.index, "columns": df.columns}

        # Save all metadata to a single pickle file
        metadata_path = os.path.join(self._temp_dir.name, "metadata.pkl")
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        # Create the context object with paths to memory-mapped files
        context = OptimizationDataContext(
            daily_data_path=paths.get("daily_data_path", ""),
            monthly_data_path=paths.get("monthly_data_path", ""),
            returns_data_path=paths.get("returns_data_path", ""),
            metadata_path=metadata_path,
            windows=data.windows,
        )

        return context

    def cleanup(self) -> None:
        """Cleans up temporary files created for memory-mapped data."""
        if self._temp_dir:
            try:
                logger.debug(f"Cleaning up shared data directory: {self._temp_dir.name}")
                self._temp_dir.cleanup()
            except Exception as e:
                logger.error(f"Failed to clean up temporary directory {self._temp_dir.name}: {e}")

    def __del__(self) -> None:
        """Ensures cleanup on object destruction."""
        self.cleanup()


def reconstruct_optimization_data(context: OptimizationDataContext) -> OptimizationData:
    """Reconstructs OptimizationData from memory-mapped files and metadata.

    Args:
        context: The context object with paths to memory-mapped files.

    Returns:
        The reconstructed OptimizationData object.
    """
    with open(context.metadata_path, "rb") as f:
        metadata = pickle.load(f)

    data_frames = {}
    for name in ["daily", "monthly", "returns"]:
        data_path = getattr(context, f"{name}_data_path")
        meta = metadata.get(name)

        if not data_path or meta is None:
            logger.debug(f"No data path or metadata for '{name}', creating empty DataFrame.")
            data_frames[name] = pd.DataFrame()
            continue

        # Load the array from the memory-mapped file
        try:
            memmapped_array = np.load(data_path, mmap_mode="r")

            # Reconstruct the DataFrame
            data_frames[name] = pd.DataFrame(
                memmapped_array, index=meta["index"], columns=meta["columns"]
            )
        except Exception as e:
            logger.error(f"Failed to load memory-mapped array from {data_path}: {e}")
            data_frames[name] = pd.DataFrame()

    return OptimizationData(
        daily=data_frames["daily"],
        monthly=data_frames["monthly"],
        returns=data_frames["returns"],
        windows=context.windows,
    )
