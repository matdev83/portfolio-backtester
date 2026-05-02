"""Research protocol primitives (configuration, scoring, structured results)."""

from .artifacts import (
    ResearchArtifactExistsError,
    ResearchArtifactWriter,
    protocol_config_to_plain,
    sanitize_scenario_name,
    write_grid_results,
    write_lock_file,
    write_selected_protocols,
    write_unseen_results,
)
from .constraints import MetricConstraint, ResearchConstraintError
from .double_oos_wfo import DoubleOOSWFOProtocol, expand_wfo_architecture_grid
from .protocol_config import (
    ArchitectureLockConfig,
    DateRangeConfig,
    DoubleOOSWFOProtocolConfig,
    ExecutionConfig,
    FinalUnseenMode,
    ReportingConfig,
    ResearchProtocolConfigError,
    SelectionConfig,
    WFOGridConfig,
    parse_double_oos_wfo_protocol,
)
from .hashing import stable_hash
from .protocol_orchestrator import ResearchProtocolOrchestrator
from .registry import (
    ResearchRegistryError,
    ResearchRunRegistry,
    compute_registry_hashes,
    unseen_period_plain,
)
from .results import (
    ResearchProtocolResult,
    SelectedProtocol,
    UnseenValidationResult,
    WFOArchitecture,
    WFOArchitectureResult,
)
from .report import generate_research_markdown_report
from .scoring import (
    ResearchScoreCalculator,
    ResearchScoringError,
    canonical_metric_display_key,
    extract_metric_value,
    select_top_protocols_by_score,
    select_top_selected_protocols,
)

__all__ = [
    "ArchitectureLockConfig",
    "DateRangeConfig",
    "DoubleOOSWFOProtocol",
    "DoubleOOSWFOProtocolConfig",
    "ExecutionConfig",
    "expand_wfo_architecture_grid",
    "FinalUnseenMode",
    "generate_research_markdown_report",
    "parse_double_oos_wfo_protocol",
    "protocol_config_to_plain",
    "MetricConstraint",
    "ReportingConfig",
    "ResearchArtifactExistsError",
    "ResearchArtifactWriter",
    "ResearchConstraintError",
    "ResearchProtocolConfigError",
    "ResearchProtocolOrchestrator",
    "ResearchProtocolResult",
    "ResearchRegistryError",
    "ResearchRunRegistry",
    "compute_registry_hashes",
    "ResearchScoreCalculator",
    "ResearchScoringError",
    "sanitize_scenario_name",
    "SelectedProtocol",
    "SelectionConfig",
    "stable_hash",
    "canonical_metric_display_key",
    "extract_metric_value",
    "select_top_protocols_by_score",
    "select_top_selected_protocols",
    "UnseenValidationResult",
    "unseen_period_plain",
    "WFOArchitecture",
    "WFOArchitectureResult",
    "WFOGridConfig",
    "write_grid_results",
    "write_lock_file",
    "write_selected_protocols",
    "write_unseen_results",
]
