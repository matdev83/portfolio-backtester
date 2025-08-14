"""
Interface definitions for portfolio backtester polymorphic components.
"""

from .parameter_extractor import ParameterExtractorFactory
from .optimization_populator import OptimizationPopulatorFactory
from .strategy_resolver import StrategyResolverFactory
from .allocation_validator import AllocationValidatorFactory
from .signal_validator import SignalValidatorFactory
from .universal_config_validator import UniversalConfigValidatorFactory
from .price_extractor_interface import PriceExtractorFactory
from .signal_generator_interface import (
    ISignalGenerator,
    SignalGeneratorFactory,
    signal_generator_factory,
)
from .series_normalizer_interface import SeriesNormalizerFactory
from .data_processor_interface import DataProcessorFactory
from .date_normalizer_interface import (
    DateNormalizerFactory as UniverseDateNormalizerFactory,
)
from .filing_date_extractor_interface import FilingDateExtractorFactory
from .holdings_extractor_interface import HoldingsExtractorFactory
from .holding_processor_interface import HoldingProcessorFactory
from .column_handler_interface import ColumnHandlerFactory, UniverseExtractorFactory
from .signal_price_extractor_interface import SignalPriceExtractorFactory
from .data_type_converter_interface import DataTypeConverterFactory
from .strategy_resolver_interface import create_strategy_resolver
from .array_converter_interface import create_array_converter
from .strategy_specification_handler import create_polymorphic_strategy_factory
from .column_type_detector import (
    create_column_type_detector,
    create_data_structure_handler,
)
from .data_validator_interface import create_data_validator
from .price_extractor_data_source import create_price_extractor
from .data_source_interface import create_data_source_factory, create_data_source
from .timeout_manager_interface import (
    create_timeout_manager_factory,
    create_timeout_manager,
)
from .cache_manager_interface import create_cache_manager_factory, create_cache_manager
from .database_loader_interface import (
    ISeedLoader,
    ILiveDBLoader,
    ILiveDBWriter,
    DatabaseLoaderFactory,
    create_seed_loader,
    create_live_db_loader,
    create_live_db_writer,
)
from .parallel_executor_interface import (
    IParallelExecutor,
    IParallelExecutorFactory,
    create_parallel_executor_factory,
    create_parallel_executor,
)
from .math_operations_interface import (
    IMathOperations,
    IMathOperationsFactory,
    create_math_operations_factory,
    create_math_operations,
)
from .timing_state_interface import (
    ITimingState,
    ITimingStateFactory,
    create_timing_state_factory,
    create_timing_state,
)
from .timing_base_interface import (
    ITimingBase,
    ITimingBaseFactory,
    create_timing_base_factory,
    create_timing_base,
)
from .time_based_timing_interface import (
    ITimeBasedTiming,
    ITimeBasedTimingFactory,
    create_time_based_timing_factory,
    create_time_based_timing,
)
from .parallel_benefit_estimator_interface import (
    IParallelBenefitEstimator,
    IParallelBenefitEstimatorFactory,
    create_parallel_benefit_estimator_factory,
    create_parallel_benefit_estimator,
)
from .attribute_accessor_interface import (
    IAttributeAccessor,
    IModuleAttributeAccessor,
    IClassAttributeAccessor,
    IObjectFieldAccessor,
    create_attribute_accessor,
    create_module_attribute_accessor,
    create_class_attribute_accessor,
    create_object_field_accessor,
)

from .validator_interface import (
    IValidator,
    IAllocationModeValidator,
    ITradeValidator,
    IModelValidator,
    IMultiValidator,
    ValidationResult,
    ValidationSeverity,
    create_validator,
    create_allocation_mode_validator,
    create_trade_validator,
    create_model_validator,
    create_composite_validator,
)
from .strategy_base_interface import (
    IStrategyBase,
    StrategyBaseAdapter,
    StrategyBaseFactory,
)

__all__ = [
    "ParameterExtractorFactory",
    "OptimizationPopulatorFactory",
    "StrategyResolverFactory",
    "AllocationValidatorFactory",
    "SignalValidatorFactory",
    "UniversalConfigValidatorFactory",
    "PriceExtractorFactory",
    "ISignalGenerator",
    "SignalGeneratorFactory",
    "signal_generator_factory",
    "SeriesNormalizerFactory",
    "DataProcessorFactory",
    "UniverseDateNormalizerFactory",
    "FilingDateExtractorFactory",
    "HoldingsExtractorFactory",
    "HoldingProcessorFactory",
    "ColumnHandlerFactory",
    "UniverseExtractorFactory",
    "SignalPriceExtractorFactory",
    "DataTypeConverterFactory",
    "create_strategy_resolver",
    "create_array_converter",
    "create_polymorphic_strategy_factory",
    "create_column_type_detector",
    "create_data_structure_handler",
    "create_data_validator",
    "create_price_extractor",
    "create_data_source_factory",
    "create_data_source",
    "create_timeout_manager_factory",
    "create_timeout_manager",
    "create_cache_manager_factory",
    "create_cache_manager",
    "ISeedLoader",
    "ILiveDBLoader",
    "ILiveDBWriter",
    "DatabaseLoaderFactory",
    "create_seed_loader",
    "create_live_db_loader",
    "create_live_db_writer",
    "IParallelExecutor",
    "IParallelExecutorFactory",
    "create_parallel_executor_factory",
    "create_parallel_executor",
    "IMathOperations",
    "IMathOperationsFactory",
    "create_math_operations_factory",
    "create_math_operations",
    "IParallelBenefitEstimator",
    "IParallelBenefitEstimatorFactory",
    "create_parallel_benefit_estimator_factory",
    "create_parallel_benefit_estimator",
    "ITimingState",
    "ITimingStateFactory",
    "create_timing_state_factory",
    "create_timing_state",
    "ITimingBase",
    "ITimingBaseFactory",
    "create_timing_base_factory",
    "create_timing_base",
    "ITimeBasedTiming",
    "ITimeBasedTimingFactory",
    "create_time_based_timing_factory",
    "create_time_based_timing",
    "IAttributeAccessor",
    "IModuleAttributeAccessor",
    "IClassAttributeAccessor",
    "IObjectFieldAccessor",
    "create_attribute_accessor",
    "create_module_attribute_accessor",
    "create_class_attribute_accessor",
    "create_object_field_accessor",
    "IStrategyBase",
    "StrategyBaseAdapter",
    "StrategyBaseFactory",
    "IValidator",
    "IAllocationModeValidator",
    "ITradeValidator",
    "IModelValidator",
    "IMultiValidator",
    "ValidationResult",
    "ValidationSeverity",
    "create_validator",
    "create_allocation_mode_validator",
    "create_trade_validator",
    "create_model_validator",
    "create_composite_validator",
]
