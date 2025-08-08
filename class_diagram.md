classDiagram
    class ConfigurationError
    class ValidationError
    class StrategyConfigSchema
    class Backtester
    class _RunBacktestNumba
    class CusipMappingDB
    class DataPreprocessingCache
    class Feature
    class FeatureFlags
    class PreparedArrays
    class _AlignedFrames
    class DummyNumba
    class ParallelWFOProcessor
    class _PreparedArraysCache
    class UniverseLoaderError
    class YamlErrorType
    class YamlError
    class YamlValidator
    class SignatureViolationError
    class ParameterViolationError
    class ReturnTypeViolationError
    class MethodSignature
    class _SignatureRecord
    class ConstraintHandler
    class _BaseOptimizerLike
    class _OptimizerProto
    class Position
    class Trade
    class PositionTracker
    class BacktestResult
    class WindowResult
    class StrategyBacktester
    class WindowEvaluator
    class BaseDataSource
    class ETFHoldingsDataSource
    class HybridDataSource
    class MemoryDataSource
    class StooqDataSource
    class YFinanceDataSource
    class ATRFeature
    class CalmarRatio
    class DPVAMS
    class SortinoRatio
    class VAMS
    class ReplacementInfo
    class AssetReplacementManager
    class ReplacementInfoResult
    class MonteCarloSimulator
    class GARCHParameters
    class AssetStatistics
    class SyntheticDataGenerator
    class ValidationResults
    class SyntheticDataValidator
    class SyntheticDataVisualInspector
    class DiversityCalculator
    class AdaptiveMutationController
    class AdaptiveCrossoverController
    class BaseOptimizer
    class EliteSolution
    class EliteArchive
    class ParallelOptimizer
    class BacktestEvaluator
    class ParameterGeneratorError
    class UnknownOptimizerError
    class OptimizerImportError
    class GeneticOptimizer
    class OptunaObjectiveAdapter
    class ProgressTracker
    class OptimizationOrchestrator
    class ParallelOptimizationRunner
    class ParameterGenerator
    class ParameterGeneratorNotInitializedError
    class ParameterGeneratorFinishedError
    class InvalidParameterSpaceError
    class ParameterEvaluationError
    class MemoryOptimizer
    class DataFrameOptimizer
    class CacheOptimizer
    class PerformanceMonitor
    class EvaluationResult
    class OptimizationResult
    class OptimizationData
    class TrialDeduplicator
    class DedupOptunaObjectiveAdapter
    class WFOWindow
    class GeneticTrial
    class GeneticParameterGenerator
    class OptunaParameterGenerator
    class BasePerformanceOptimizer
    class BaseTrialDeduplicator
    class GenericTrialDeduplicator
    class DeduplicationFactory
    class PerformanceOptimizerFactory
    class GenericPerformanceOptimizer
    class OptunaPerformanceOptimizer
    class GeneticPerformanceOptimizer
    class GeneticTrialDeduplicator
    class GeneticParallelRunner
    class AbstractPerformanceOptimizer
    class AbstractTradeTracker
    class AbstractTrialDeduplicator
    class AbstractParallelRunner
    class AbstractPerformanceOptimizerFactory
    class OptunaTrialDeduplicator
    class OptunaParallelRunner
    class OptunaDedupObjectiveWrapper
    class BaseParallelRunner
    class GenericParallelRunner
    class VectorizedTradeTracker
    class BasePositionSizer
    class EqualWeightSizer
    class RollingSharpeSizer
    class RollingSortinoSizer
    class RollingBetaSizer
    class RollingBenchmarkCorrSizer
    class RollingDownsideVolatilitySizer
    class OptimizerReportGenerator
    class BaseStopLoss
    class NoStopLoss
    class AtrBasedStopLoss
    class BaseRoRoSignal
    class DummyRoRoSignal
    class StrategyFactory
    class BaseStrategy
    class MetaStrategyReporter
    class SubStrategyAllocation
    class BaseMetaStrategy
    class PortfolioStrategy
    class PortfolioValueTracker
    class SignalStrategy
    
    %% Provider Interface System (New Architecture)
    class IUniverseProvider
    class ConfigBasedUniverseProvider
    class FixedListUniverseProvider
    class DynamicUniverseProvider
    class UniverseProviderFactory
    class IPositionSizerProvider
    class ConfigBasedPositionSizerProvider
    class FixedPositionSizerProvider
    class PositionSizerProviderFactory
    class IStopLossProvider
    class ConfigBasedStopLossProvider
    class FixedStopLossProvider
    class StopLossProviderFactory
    class TradeAggregator
    class MetaStrategyTradeInterceptor
    class TradeSide
    class TradeRecord
    class PositionRecord
    class StopLossTesterStrategy
    class SimpleMetaStrategy
    class CalmarMomentumPortfolioStrategy
    class FilteredLaggedMomentumPortfolioStrategy
    class FixedWeightPortfolioStrategy
    class LowVolatilityFactorPortfolioStrategy
    class MomentumBetaFilteredStrategy
    class MomentumDvolSizerStrategy
    class MomentumStrategy
    class MomentumUnfilteredAtrStrategy
    class SharpeMomentumStrategy
    class SortinoMomentumStrategy
    class VAMSMomentumStrategy
    class VAMSNoDownsideStrategy
    class VolatilityTargetedFixedWeightPortfolioStrategy
    class DummyStrategyForTesting
    class EMAStrategy
    class EMARoRoStrategy
    class SeasonalSignalStrategy
    class UvxyRsiStrategy
    class TimingConfigValidator
    class TimingConfigSchema
    class CustomTimingRegistry
    class TimingControllerFactory
    class AdaptiveTimingController
    class MomentumTimingController
    class SignalBasedTiming
    class TimeBasedTiming
    class TimingController
    class TimingLogEntry
    class TimingLogger
    class PositionInfo
    class TimingState
    class TradeTracker
    class TransactionCostModel
    class RealisticTransactionCostModel
    class TradeCommissionInfo
    class UnifiedCommissionCalculator
    class _EdgarCompanyProto
    class TimeoutManager
    main --|> Backtester
    main --|> ConfigurationError
    main --|> YamlValidator
    load_scenario_from_file --|> ConfigurationError
    load_scenario_from_file --|> YamlValidator
    load_config --|> ConfigurationError
    load_config --|> YamlValidator
    validate_config_files --|> YamlValidator
    load_globals_only --|> ConfigurationError
    StrategyConfigSchema --|> ValidationError
    Backtester --|> TimeoutManager
    Backtester --|> StrategyBacktester
    Backtester --|> BacktestEvaluator
    Backtester --|> AssetReplacementManager
    Backtester --|> OptimizationData
    Backtester --|> OptimizationOrchestrator
    Backtester --|> ParallelOptimizationRunner
    get_global_cache --|> DataPreprocessingCache
    prepare_ndarrays --|> PreparedArrays
    create_parallel_wfo_processor --|> ParallelWFOProcessor
    get_or_prepare --|> PreparedArrays
    _ensure_mapping --|> YamlError
    _validate_common_keys --|> YamlError
    _resolve_strategy_and_tunables --|> YamlError
    _validate_common_types_and_windows --|> YamlError
    _validate_optimizers_section --|> YamlError
    _validate_strategy_params --|> YamlError
    _validate_universe_config --|> YamlError
    _validate_meta_strategy_logic --|> YamlError
    _validate_meta_strategy_allocations --|> YamlError
    _validate_portfolio_strategy_logic --|> YamlError
    _validate_signal_strategy_logic --|> YamlError
    _validate_diagnostic_strategy_logic --|> YamlError
    _validate_timing_config --|> YamlError
    _validate_universal_configuration_logic --|> YamlError
    validate_scenario_semantics --|> YamlError
    validate_scenario_file --|> YamlValidator
    _parse_universe_file --|> UniverseLoaderError
    load_named_universe --|> UniverseLoaderError
    lint_files --|> YamlValidator
    YamlValidator --|> YamlError
    validate_yaml_file --|> YamlValidator
    _validate_parameters --|> ParameterViolationError
    _validate_return_type --|> ReturnTypeViolationError
    register_method --|> MethodSignature
    handle_constraints --|> ConstraintHandler
    get_data_source --|> HybridDataSource
    get_data_source --|> MemoryDataSource
    run_backtest_mode --|> StrategyBacktester
    get_optimizer --|> GeneticOptimizer
    calculate_portfolio_returns --|> TradeTracker
    _create_meta_strategy_trade_tracker --|> TradeTracker
    PositionTracker --|> Position
    PositionTracker --|> Trade
    StrategyBacktester --|> WindowResult
    StrategyBacktester --|> BacktestResult
    WindowEvaluator --|> WindowResult
    WindowEvaluator --|> PositionTracker
    HybridDataSource --|> YFinanceDataSource
    HybridDataSource --|> StooqDataSource
    AssetReplacementManager --|> ReplacementInfoResult
    AssetReplacementManager --|> ReplacementInfo
    AssetReplacementManager --|> SyntheticDataGenerator
    SyntheticDataGenerator --|> AssetStatistics
    SyntheticDataGenerator --|> GARCHParameters
    SyntheticDataValidator --|> ValidationResults
    SyntheticDataVisualInspector --|> SyntheticDataGenerator
    EliteArchive --|> EliteSolution
    ParallelOptimizer --|> ParallelOptimizer
    BacktestEvaluator --|> WindowResult
    BacktestEvaluator --|> EvaluationResult
    BacktestEvaluator --|> WFOWindow
    BacktestEvaluator --|> WindowEvaluator
    create_parameter_generator --|> OptimizerImportError
    create_parameter_generator --|> UnknownOptimizerError
    create_parameter_generator --|> ParameterGeneratorError
    _create_optuna_generator --|> OptimizerImportError
    _create_optuna_generator --|> OptunaParameterGenerator
    _create_genetic_generator --|> GeneticParameterGenerator
    _create_genetic_generator --|> OptimizerImportError
    OptunaObjectiveAdapter --|> StrategyBacktester
    OptunaObjectiveAdapter --|> BacktestEvaluator
    ProgressTracker --|> ProgressTracker
    OptimizationOrchestrator --|> OptimizationResult
    _optuna_worker --|> OptunaObjectiveAdapter
    ParallelOptimizationRunner --|> OptimizationResult
    validate_parameter_space --|> InvalidParameterSpaceError
    validate_optimization_config --|> ParameterGeneratorError
    MemoryOptimizer --|> MemoryOptimizer
    TrialDeduplicator --|> TrialDeduplicator
    create_deduplicating_objective --|> DedupOptunaObjectiveAdapter
    GeneticParameterGenerator --|> OptimizationResult
    GeneticParameterGenerator --|> ParameterGeneratorNotInitializedError
    GeneticParameterGenerator --|> ParameterGeneratorFinishedError
    GeneticParameterGenerator --|> GeneticTrial
    GeneticParameterGenerator --|> InvalidParameterSpaceError
    OptunaParameterGenerator --|> OptimizationResult
    OptunaParameterGenerator --|> ParameterGeneratorNotInitializedError
    OptunaParameterGenerator --|> ParameterGeneratorFinishedError
    OptunaParameterGenerator --|> ParameterGeneratorError
    DeduplicationFactory --|> GenericTrialDeduplicator
    PerformanceOptimizerFactory --|> GenericPerformanceOptimizer
    PerformanceOptimizerFactory --|> OptunaPerformanceOptimizer
    PerformanceOptimizerFactory --|> GeneticPerformanceOptimizer
    GenericPerformanceOptimizer --|> VectorizedTradeTracker
    GenericPerformanceOptimizer --|> GenericTrialDeduplicator
    GenericPerformanceOptimizer --|> GenericParallelRunner
    GeneticPerformanceOptimizer --|> GeneticTrialDeduplicator
    GeneticPerformanceOptimizer --|> GeneticParallelRunner
    GeneticPerformanceOptimizer --|> VectorizedTradeTracker
    GeneticPerformanceOptimizer --|> GenericTrialDeduplicator
    OptunaPerformanceOptimizer --|> OptunaParallelRunner
    OptunaPerformanceOptimizer --|> VectorizedTradeTracker
    OptunaPerformanceOptimizer --|> OptunaTrialDeduplicator
    OptunaPerformanceOptimizer --|> BaseTrialDeduplicator
    OptunaParallelRunner --|> ParallelOptimizationRunner
    create_optimization_report --|> OptimizerReportGenerator
    AtrBasedStopLoss --|> ATRFeature
    BaseStrategy --|> SignalBasedTiming
    BaseStrategy --|> TimeBasedTiming
    
    %% Provider Interface Relationships (New Architecture)
    BaseStrategy --|> IUniverseProvider
    BaseStrategy --|> IPositionSizerProvider
    BaseStrategy --|> IStopLossProvider
    ConfigBasedUniverseProvider --|> IUniverseProvider
    FixedListUniverseProvider --|> IUniverseProvider
    DynamicUniverseProvider --|> IUniverseProvider
    ConfigBasedPositionSizerProvider --|> IPositionSizerProvider
    FixedPositionSizerProvider --|> IPositionSizerProvider
    ConfigBasedStopLossProvider --|> IStopLossProvider
    FixedStopLossProvider --|> IStopLossProvider
    UniverseProviderFactory --|> IUniverseProvider
    PositionSizerProviderFactory --|> IPositionSizerProvider
    StopLossProviderFactory --|> IStopLossProvider
    PortfolioStrategy --|> BaseStrategy
    SignalStrategy --|> BaseStrategy
    
    BaseMetaStrategy --|> TradeAggregator
    BaseMetaStrategy --|> PortfolioValueTracker
    BaseMetaStrategy --|> MetaStrategyTradeInterceptor
    BaseMetaStrategy --|> SubStrategyAllocation
    BaseMetaStrategy --|> MetaStrategyReporter
    BaseMetaStrategy --|> TradeRecord
    PortfolioValueTracker --|> PositionRecord
    TradeAggregator --|> PositionRecord
    MetaStrategyTradeInterceptor --|> TradeRecord
    TradeRecord --|> TradeSide
    CalmarMomentumPortfolioStrategy --|> CalmarRatio
    LowVolatilityFactorPortfolioStrategy --|> ETFHoldingsDataSource
    SortinoMomentumStrategy --|> SortinoRatio
    VAMSMomentumStrategy --|> DPVAMS
    DummyStrategyForTesting --|> NoStopLoss
    DummyStrategyForTesting --|> AtrBasedStopLoss
    TimingConfigSchema --|> ValidationError
    TimingControllerFactory --|> SignalBasedTiming
    TimingControllerFactory --|> TimeBasedTiming
    AdaptiveTimingController --|> TimeBasedTiming
    TimingController --|> TimingState
    TimingLogger --|> TimingLogEntry
    get_timing_logger --|> TimingLogger
    configure_timing_logging --|> TimingLogger
    TimingState --|> PositionInfo
    track_trades_vectorized --|> VectorizedTradeTracker
    TradeTracker --|> Trade
    get_transaction_cost_model --|> RealisticTransactionCostModel
    UnifiedCommissionCalculator --|> TradeCommissionInfo
    get_unified_commission_calculator --|> UnifiedCommissionCalculator
    generate_enhanced_wfo_windows --|> WFOWindow
