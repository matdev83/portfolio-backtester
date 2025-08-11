from typing import Any, Dict, Optional
from portfolio_backtester.optimization.parallel_optimization_runner import (
    ParallelOptimizationRunner,
)
from portfolio_backtester.optimization.results import OptimizationData


class MockParallelOptimizationRunner(ParallelOptimizationRunner):
    """
    A mock version of the ParallelOptimizationRunner that allows direct access
    to its protected methods for unit testing.
    """

    def __init__(
        self,
        data: OptimizationData,
        scenario_config: Optional[Dict[str, Any]] = None,
        optimization_config: Optional[Dict[str, Any]] = None,
    ):
        # Call the real __init__ with minimal required arguments
        super().__init__(
            scenario_config=scenario_config or {},
            optimization_config=optimization_config or {},
            data=data,
        )
