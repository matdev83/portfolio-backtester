"""
GPU acceleration for genetic algorithm optimization.

This module provides utilities to leverage GPU acceleration for fitness evaluation
in strategies with vectorizable operations, using CuPy for CUDA-based computations.
"""

# mypy: ignore-errors
# The above directive is needed because we're conditionally importing cupy
# which might not be installed, so mypy can't verify its types

import time
import importlib.util
from typing import Any, Dict, List, Optional
import numpy as np
from loguru import logger


class GPUAccelerationManager:
    """
    Manages GPU acceleration for fitness evaluation.

    This class provides utilities to detect GPUs, manage memory, and accelerate
    vectorizable operations using CuPy when available.
    """

    def __init__(self, gpu_device: int = 0, memory_fraction: float = 0.8):
        """
        Initialize the GPU acceleration manager.

        Args:
            gpu_device: GPU device ID to use (0 = first GPU)
            memory_fraction: Fraction of GPU memory to use (0.0-1.0)
        """
        self.gpu_device = gpu_device
        self.memory_fraction = min(1.0, max(0.1, memory_fraction))

        # State
        self._cupy_available = False
        self._initialized = False
        self._xp = np  # Default to numpy

        # Try to import cupy
        try:
            if importlib.util.find_spec("cupy") is not None:
                import cupy as cp

                self._cupy_available = True
                self._xp = cp
                logger.debug("CuPy available for GPU acceleration")
            else:
                logger.debug("CuPy not found, using NumPy for calculations")
        except ImportError:
            logger.debug("Failed to import CuPy, using NumPy for calculations")

    def initialize(self) -> bool:
        """
        Initialize GPU resources.

        Returns:
            True if GPU acceleration is available and initialized, False otherwise
        """
        if not self._cupy_available:
            return False

        if self._initialized:
            return True

        try:
            import cupy as cp

            # Set the current device
            cp.cuda.Device(self.gpu_device).use()

            # Check if memory pool is available
            if hasattr(cp.cuda, "MemoryPool"):
                pool = cp.cuda.MemoryPool()
                cp.cuda.set_allocator(pool.malloc)
                logger.debug(f"CuPy memory pool initialized for device {self.gpu_device}")

                # Limit memory usage if supported
                if hasattr(pool, "set_limit"):
                    try:
                        # Get total GPU memory
                        device = cp.cuda.Device(self.gpu_device)
                        total_memory = device.attributes["TotalGlobalMem"]
                        # Set limit to fraction of total memory
                        limit = int(total_memory * self.memory_fraction)
                        pool.set_limit(size=limit)
                        logger.debug(
                            f"Memory limit set to {self.memory_fraction:.1%} of GPU memory"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to set memory limit: {e}")

            self._initialized = True
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize GPU acceleration: {e}")
            self._cupy_available = False
            self._xp = np
            return False

    def is_available(self) -> bool:
        """
        Check if GPU acceleration is available.

        Returns:
            True if GPU acceleration is available, False otherwise
        """
        return self._cupy_available

    def get_array_module(self) -> Any:
        """
        Get the array module to use (NumPy or CuPy).

        This follows the same pattern as CuPy's get_array_module function.

        Returns:
            NumPy or CuPy module
        """
        return self._xp

    def to_gpu(self, arr: Any) -> Any:
        """
        Transfer a NumPy array to the GPU.

        Args:
            arr: NumPy array to transfer

        Returns:
            CuPy array on GPU or original NumPy array if GPU acceleration is not available
        """
        if not self._cupy_available or not self._initialized:
            return arr

        try:
            import cupy as cp

            return cp.asarray(arr)
        except Exception as e:
            logger.warning(f"Failed to transfer array to GPU: {e}")
            return arr

    def to_cpu(self, arr: Any) -> Any:
        """
        Transfer a GPU array back to CPU.

        Args:
            arr: GPU array to transfer

        Returns:
            NumPy array on CPU
        """
        if not self._cupy_available or not self._initialized:
            return arr

        try:
            import cupy as cp

            if isinstance(arr, cp.ndarray):
                return cp.asnumpy(arr)
            return arr
        except Exception as e:
            logger.warning(f"Failed to transfer array to CPU: {e}")
            return arr

    def synchronize(self) -> None:
        """Synchronize the GPU device (wait for all operations to complete)."""
        if not self._cupy_available or not self._initialized:
            return

        try:
            import cupy as cp

            cp.cuda.Stream.null.synchronize()
        except Exception as e:
            logger.warning(f"Failed to synchronize GPU: {e}")

    def benchmark(self, sizes: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Benchmark CPU vs GPU performance for typical operations.

        Args:
            sizes: List of matrix sizes to benchmark (defaults to [1000, 2000, 4000])

        Returns:
            Dictionary with benchmark results
        """
        if sizes is None:
            sizes = [1000, 2000, 4000]

        results: Dict[str, Any] = {
            "gpu_available": self._cupy_available,
            "sizes": sizes,
            "cpu_times": [],
            "gpu_times": [],
            "speedups": [],
        }

        if not self._cupy_available:
            logger.warning("GPU acceleration not available, skipping benchmark")
            return results

        try:
            import cupy as cp

            for size in sizes:
                # Create random matrices
                a_cpu = np.random.random((size, size)).astype(np.float32)
                b_cpu = np.random.random((size, size)).astype(np.float32)

                # CPU benchmark
                start_time = time.time()
                _ = np.dot(a_cpu, b_cpu)  # Perform computation but result not needed
                cpu_time = time.time() - start_time

                # GPU benchmark
                a_gpu = cp.asarray(a_cpu)
                b_gpu = cp.asarray(b_cpu)

                # Warm-up
                _ = cp.dot(a_gpu, b_gpu)  # Perform computation but result not needed
                cp.cuda.Stream.null.synchronize()

                # Actual benchmark
                start_time = time.time()
                _ = cp.dot(a_gpu, b_gpu)  # Perform computation but result not needed
                cp.cuda.Stream.null.synchronize()
                gpu_time = time.time() - start_time

                # Calculate speedup
                speedup = cpu_time / max(gpu_time, 1e-6)

                results["cpu_times"].append(cpu_time)
                results["gpu_times"].append(gpu_time)
                results["speedups"].append(speedup)

                logger.debug(
                    f"Size {size}x{size}: CPU {cpu_time:.4f}s, GPU {gpu_time:.4f}s, "
                    f"Speedup: {speedup:.2f}x"
                )

            return results
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return results

    def accelerate_evaluation(
        self,
        prices: np.ndarray,
        weights: np.ndarray,
        returns: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Accelerate fitness evaluation computations using GPU.

        This method adapts the calculation path based on whether GPU is available.

        Args:
            prices: Price data array
            weights: Weight data array
            returns: Returns data array

        Returns:
            Dictionary with accelerated computation results
        """
        if not self._cupy_available or not self._initialized:
            # Use CPU path with NumPy
            return self._calculate_on_cpu(prices, weights, returns)

        try:
            # Use GPU path with CuPy
            return self._calculate_on_gpu(prices, weights, returns)
        except Exception as e:
            logger.warning(f"GPU calculation failed, falling back to CPU: {e}")
            return self._calculate_on_cpu(prices, weights, returns)

    def _calculate_on_cpu(
        self,
        prices: np.ndarray,
        weights: np.ndarray,
        returns: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics on CPU using NumPy.

        Args:
            prices: Price data array
            weights: Weight data array
            returns: Returns data array

        Returns:
            Dictionary with computation results
        """
        # Example calculation (would be replaced with actual calculations)
        portfolio_returns = np.sum(weights * returns, axis=1)

        # Calculate cumulative returns
        cum_returns = np.cumprod(1 + portfolio_returns) - 1

        # Calculate drawdowns
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / (peak + 1)
        max_drawdown = np.min(drawdown)

        # Calculate Sharpe ratio (assuming daily returns)
        sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)

        # Calculate Calmar ratio
        calmar = np.mean(portfolio_returns) * 252 / abs(max_drawdown) if max_drawdown < 0 else 0

        return {
            "portfolio_returns": portfolio_returns,
            "cum_returns": cum_returns,
            "drawdown": drawdown,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "calmar": calmar,
            "used_gpu": False,
        }

    def _calculate_on_gpu(
        self,
        prices: np.ndarray,
        weights: np.ndarray,
        returns: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics on GPU using CuPy.

        Args:
            prices: Price data array
            weights: Weight data array
            returns: Returns data array

        Returns:
            Dictionary with computation results
        """
        import cupy as cp

        # Transfer data to GPU
        # Note: prices data not used in this example calculation, but would be in real use
        _ = cp.asarray(prices)  # Transfer to GPU even if not used, to simulate full workload
        gpu_weights = cp.asarray(weights)
        gpu_returns = cp.asarray(returns)

        # Example calculation (would be replaced with actual calculations)
        gpu_portfolio_returns = cp.sum(gpu_weights * gpu_returns, axis=1)

        # Calculate cumulative returns
        gpu_cum_returns = cp.cumprod(1 + gpu_portfolio_returns) - 1

        # Calculate drawdowns
        gpu_peak = cp.maximum.accumulate(gpu_cum_returns)
        gpu_drawdown = (gpu_cum_returns - gpu_peak) / (gpu_peak + 1)
        gpu_max_drawdown = cp.min(gpu_drawdown)

        # Calculate Sharpe ratio (assuming daily returns)
        gpu_sharpe = cp.mean(gpu_portfolio_returns) / cp.std(gpu_portfolio_returns) * cp.sqrt(252)

        # Calculate Calmar ratio
        gpu_calmar = (
            cp.mean(gpu_portfolio_returns) * 252 / abs(float(gpu_max_drawdown))
            if gpu_max_drawdown < 0
            else 0
        )

        # Transfer results back to CPU
        result = {
            "portfolio_returns": cp.asnumpy(gpu_portfolio_returns),
            "cum_returns": cp.asnumpy(gpu_cum_returns),
            "drawdown": cp.asnumpy(gpu_drawdown),
            "max_drawdown": float(gpu_max_drawdown),
            "sharpe": float(gpu_sharpe),
            "calmar": float(gpu_calmar),
            "used_gpu": True,
        }

        return result

    def cleanup(self) -> None:
        """Clean up GPU resources."""
        if not self._cupy_available or not self._initialized:
            return

        try:
            import cupy as cp

            cp.get_default_memory_pool().free_all_blocks()
            logger.debug("GPU memory pool cleared")
        except Exception as e:
            logger.warning(f"Failed to clean up GPU resources: {e}")

    def __del__(self) -> None:
        """Clean up resources when the object is destroyed."""
        self.cleanup()
