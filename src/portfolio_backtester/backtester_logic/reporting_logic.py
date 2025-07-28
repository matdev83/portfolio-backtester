import pandas as pd
from ..reporting.optimizer_report_generator import create_optimization_report
from ..reporting.performance_metrics import calculate_metrics
import logging

logger = logging.getLogger(__name__)

def generate_optimization_report(backtester, scenario_config, optimal_params, full_rets, best_trial_obj, actual_num_trials):
    """Generate comprehensive optimization report with performance analysis."""
    
    strategy_name = scenario_config["name"]
    advanced_reporting_config = backtester.global_config.get('advanced_reporting_config', {})
    
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Generating comprehensive optimization report for {strategy_name}")
    
    defer_expensive_plots = advanced_reporting_config.get('defer_expensive_plots', True)
    defer_parameter_analysis = advanced_reporting_config.get('defer_parameter_analysis', True)
    
    if defer_expensive_plots:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("⚡ Using performance-optimized reporting (expensive plots deferred)")
    else:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("📊 Generating full comprehensive report (may take longer)")
    
    if full_rets is not None and not full_rets.empty:
        benchmark_ticker = backtester.global_config["benchmark"]
        
        if hasattr(backtester, 'daily_data_ohlc') and backtester.daily_data_ohlc is not None:
            if isinstance(backtester.daily_data_ohlc.columns, pd.MultiIndex):
                benchmark_data = backtester.daily_data_ohlc.xs(benchmark_ticker, level='Ticker', axis=1)['Close']
            else:
                benchmark_data = backtester.daily_data_ohlc[benchmark_ticker]
            
            benchmark_aligned = benchmark_data.reindex(full_rets.index)
            benchmark_returns = benchmark_aligned.pct_change(fill_method=None).fillna(0)
        else:
            benchmark_returns = pd.Series(0.0, index=full_rets.index)
        
        performance_metrics = calculate_metrics(full_rets, benchmark_returns, benchmark_ticker)
    else:
        logger.warning("No returns data available for performance metrics calculation")
        performance_metrics = {}
    
    optimization_results = {
        "strategy_name": strategy_name,
        "optimal_parameters": optimal_params,
        "performance_metrics": performance_metrics,
        "optimization_metadata": {
            "num_trials": actual_num_trials,
            "optimizer_type": getattr(backtester.args, "optimizer", "optuna"),
            "optimization_date": pd.Timestamp.now().isoformat(),
            "global_config": backtester.global_config,
            "defer_expensive_plots": defer_expensive_plots,
            "defer_parameter_analysis": defer_parameter_analysis
        }
    }
    
    if best_trial_obj and hasattr(best_trial_obj, 'study') and not defer_parameter_analysis:
        try:
            study = best_trial_obj.study
            trials_data = []
            
            for trial in study.trials:
                trial_data = {
                    "number": trial.number,
                    "value": trial.value if trial.value is not None else float('nan'),
                    "params": trial.params,
                    "state": trial.state.name,
                }
                
                if trial.user_attrs:
                    trial_data["user_attrs"] = trial.user_attrs
                
                trials_data.append(trial_data)
            
            optimization_results["trials_data"] = trials_data
            optimization_results["best_trial_number"] = best_trial_obj.number
            
            if len(trials_data) > 10:
                try:
                    import optuna
                    param_importance = optuna.importance.get_param_importances(study)
                    optimization_results["parameter_importance"] = param_importance
                except Exception as e:
                    logger.warning(f"Could not calculate parameter importance: {e}")
            
        except Exception as e:
            logger.warning(f"Could not extract trial data from study: {e}")
    elif defer_parameter_analysis:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("⚡ Parameter analysis deferred for performance")
    
    optimized_result_name = list(backtester.results.keys())[-1]
    result_data = backtester.results[optimized_result_name]
    
    additional_info = {
        "num_trials": actual_num_trials,
        "best_trial_number": best_trial_obj.number if best_trial_obj else None,
        "optimization_time": "Not tracked",
        "random_seed": getattr(backtester, 'random_state', None),
        "constraint_info": {
            "status": result_data.get("constraint_status", "UNKNOWN"),
            "message": result_data.get("constraint_message", ""),
            "violations": result_data.get("constraint_violations", []),
            "constraints_config": result_data.get("constraints_config", [])
        },
        "performance_optimizations": {
            "defer_expensive_plots": defer_expensive_plots,
            "defer_parameter_analysis": defer_parameter_analysis
        }
    }
    
    try:
        report_path = create_optimization_report(
            strategy_name=strategy_name,
            optimization_results=optimization_results,
            performance_metrics=performance_metrics,
            optimal_parameters=optimal_params,
            plots_source_dir="plots",
            run_id=None,
            additional_info=additional_info
        )
        
        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Comprehensive optimization report generated: {report_path}")
        logger.info(f"Optimization Report Generated: {report_path}")
        logger.info("Report directory contains:")
        logger.info("   - optimization_report.md (Main report)")
        logger.info("   - plots/ (All generated visualizations)")
        logger.info("   - data/ (Raw optimization data)")
        
    except Exception as e:
        logger.error(f"Failed to create optimization report: {e}")
        raise
