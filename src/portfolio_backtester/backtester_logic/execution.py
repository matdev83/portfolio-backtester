import pandas as pd
from ..utils import INTERRUPTED as CENTRAL_INTERRUPTED_FLAG
from ..reporting.optimizer_report_generator import create_optimization_report
from ..reporting.performance_metrics import calculate_metrics
from .constraint_handler import ConstraintHandler
import logging

logger = logging.getLogger(__name__)

def run_backtest_mode(self, scenario_config, monthly_data, daily_data, rets_full):
    if self.logger.isEnabledFor(logging.INFO):
        self.logger.info(f"Running backtest for scenario: {scenario_config['name']}")
    
    if self.args.study_name:
        try:
            import optuna
            study = optuna.load_study(study_name=self.args.study_name, storage="sqlite:///optuna_studies.db")
            optimal_params = scenario_config["strategy_params"].copy()
            optimal_params.update(study.best_params)
            scenario_config["strategy_params"] = optimal_params
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f"Loaded best parameters from study '{self.args.study_name}': {optimal_params}")
        except KeyError:
            self.logger.warning(f'Study \'{self.args.study_name}\' not found. Using default parameters for scenario \'{scenario_config["name"]}\'.')
        except Exception as e:
            self.logger.error(f"Error loading Optuna study: {e}. Using default parameters.")

    rets = self.run_scenario(scenario_config, monthly_data, daily_data, rets_full)
    train_end_date = pd.to_datetime(scenario_config.get("train_end_date", "2018-12-31"))
    self.results[scenario_config["name"]] = {"returns": rets, "display_name": scenario_config["name"], "train_end_date": train_end_date}

def run_optimize_mode(self, scenario_config, monthly_data, daily_data, rets_full):
    if self.logger.isEnabledFor(logging.INFO):
        self.logger.info(f"Running optimization for scenario: {scenario_config['name']}")
    
    optimal_params, actual_num_trials, best_trial_obj = self.run_optimization(scenario_config, monthly_data, daily_data, rets_full)
    
    if CENTRAL_INTERRUPTED_FLAG:
        self.logger.warning(f"Optimization for {scenario_config['name']} was interrupted. Skipping full backtest with potentially incomplete optimal parameters.")
        interrupted_name = f'{scenario_config["name"]} (Optimization Interrupted)'
        self.results[interrupted_name] = {
            "returns": pd.Series(dtype=float),
            "display_name": interrupted_name,
            "num_trials_for_dsr": actual_num_trials if actual_num_trials is not None else 0,
            "train_end_date": pd.to_datetime(scenario_config.get("train_end_date", "2018-12-31")),
            "notes": "Optimization process was interrupted by the user."
        }
        return
    
    if optimal_params is None:
        self.logger.error(f"Optimization for {scenario_config['name']} did not yield optimal parameters. Skipping full backtest.")
        return

    optimized_scenario = scenario_config.copy()
    optimized_scenario["strategy_params"] = optimal_params
    
    if self.logger.isEnabledFor(logging.INFO):
        self.logger.info(f"Running full backtest with optimal parameters: {optimal_params}")
    full_rets = self.run_scenario(optimized_scenario, monthly_data, daily_data, rets_full)
    
    # Initialize optimized_name with default value
    optimized_name = f'{scenario_config["name"]} (Optimized)'
    
    # Initialize constraint-related variables with defaults
    constraint_status = "NO_CONSTRAINTS"
    constraint_message = "No constraints defined"
    constraint_violations = []
    
    # Validate constraints on final backtest results
    constraints_config = scenario_config.get("optimization_constraints", [])
    if constraints_config and full_rets is not None and not full_rets.empty:
        # Calculate metrics for constraint validation
        benchmark_ticker = self.global_config["benchmark"]
        if hasattr(self, 'daily_data_ohlc') and self.daily_data_ohlc is not None:
            if isinstance(self.daily_data_ohlc.columns, pd.MultiIndex):
                benchmark_data = self.daily_data_ohlc.xs(benchmark_ticker, level='Ticker', axis=1)['Close']
            else:
                benchmark_data = self.daily_data_ohlc[benchmark_ticker]
            
            # Ensure benchmark_data is aligned with full_rets before calculating returns
            benchmark_aligned = benchmark_data.reindex(full_rets.index)
            benchmark_returns = benchmark_aligned.pct_change(fill_method=None).fillna(0)
            
            final_metrics = calculate_metrics(full_rets, benchmark_returns, benchmark_ticker)
            
            # Check each constraint
            constraint_violations = []
            for constraint in constraints_config:
                metric_name = constraint.get("metric")
                min_value = constraint.get("min_value")
                max_value = constraint.get("max_value")
                
                if metric_name and (min_value is not None or max_value is not None):
                    metric_val = final_metrics.get(metric_name)
                    
                    if metric_val is not None and not pd.isna(metric_val):
                        violated = False
                        violation_msg = ""
                        
                        if min_value is not None and metric_val < min_value:
                            violated = True
                            violation_msg = f"{metric_name} = {metric_val:.4f} < {min_value} (min)"
                        if max_value is not None and metric_val > max_value:
                            violated = True
                            violation_msg = f"{metric_name} = {metric_val:.4f} > {max_value} (max)"
                        
                        if violated:
                            constraint_violations.append(violation_msg)
                            self.logger.warning(f"⚠️  CONSTRAINT VIOLATION: {violation_msg}")
            
            if constraint_violations:
                # Add highly visible constraint violation warnings
                violation_banner = "=" * 80
                self.logger.error(f"\n{violation_banner}")
                self.logger.error(f"🚨 CONSTRAINT VIOLATION DETECTED! 🚨")
                self.logger.error(f"{violation_banner}")
                self.logger.error(f"Final backtest violates {len(constraint_violations)} constraint(s)!")
                self.logger.error("Attempting to find constraint-satisfying parameters...")
                for i, violation in enumerate(constraint_violations, 1):
                    self.logger.error(f"  {i}. {violation}")
                self.logger.error(f"{violation_banner}")
                
                # Also print to console for immediate visibility
                print(f"\n{violation_banner}")
                print(f"🚨 CONSTRAINT VIOLATION DETECTED! 🚨")
                print(f"{violation_banner}")
                print(f"❌ Optimization failed to satisfy constraints:")
                for i, violation in enumerate(constraint_violations, 1):
                    print(f"   {i}. {violation}")
                print(f"{violation_banner}")
                
                # Try to find constraint-satisfying parameters using the constraint handler
                constraint_handler = ConstraintHandler(self.global_config)
                
                adjusted_params, adjusted_rets, success = constraint_handler.find_constraint_satisfying_params(
                    scenario_config=scenario_config,
                    optimal_params=optimal_params,
                    constraint_violations=constraint_violations,
                    monthly_data=monthly_data,
                    daily_data=daily_data,
                    rets_full=rets_full,
                    run_scenario_func=self.run_scenario,
                    benchmark_returns=benchmark_returns,
                    benchmark_ticker=benchmark_ticker,
                    max_attempts=5
                )
                
                if success and adjusted_rets is not None:
                    if self.logger.isEnabledFor(logging.INFO):
                        self.logger.info("✅ Successfully found constraint-satisfying parameters!")
                    print("✅ Successfully found constraint-satisfying parameters!")
                    optimized_name = f'{scenario_config["name"]} (Constraint-Adjusted)'
                    full_rets = adjusted_rets
                    optimal_params = adjusted_params
                    constraint_status = "ADJUSTED"
                    constraint_message = "Constraints satisfied after parameter adjustment"
                else:
                    # Fall back to default parameters as last resort
                    self.logger.warning("Could not find constraint-satisfying parameters. Falling back to defaults.")
                    print("⚠️  Could not find constraint-satisfying parameters. Using fallback defaults.")
                    fallback_scenario = scenario_config.copy()
                    fallback_scenario["strategy_params"] = scenario_config["strategy_params"].copy()
                    
                    fallback_rets = self.run_scenario(fallback_scenario, monthly_data, daily_data, rets_full)
                    
                    if fallback_rets is not None and not fallback_rets.empty:
                        fallback_metrics = calculate_metrics(fallback_rets, benchmark_returns, benchmark_ticker)
                        fallback_violations = constraint_handler._check_constraint_violations(fallback_metrics, constraints_config)
                        
                        if fallback_violations:
                            self.logger.error("Even fallback parameters violate constraints!")
                            print("❌ Even fallback parameters violate constraints!")
                            print("💡 RECOMMENDATION: Relax constraints or modify strategy configuration")
                            optimized_name = f'{scenario_config["name"]} (Constraint Violation - Fallback)'
                            constraint_status = "VIOLATED"
                            constraint_message = f"All parameter combinations violate constraints. Violations: {', '.join(fallback_violations)}"
                        else:
                            if self.logger.isEnabledFor(logging.INFO):
                                self.logger.info("✅ Fallback parameters satisfy constraints.")
                            print("✅ Fallback parameters satisfy constraints.")
                            optimized_name = f'{scenario_config["name"]} (Fallback - Constraints OK)'
                            constraint_status = "FALLBACK_OK"
                            constraint_message = "Constraints satisfied using fallback parameters"
                        
                        full_rets = fallback_rets
                        optimal_params = fallback_scenario["strategy_params"]
                    else:
                        self.logger.error("Fallback backtest failed. Using original violating result.")
                        print("❌ Fallback backtest failed. Using original violating result.")
                        optimized_name = f'{scenario_config["name"]} (Constraint Violation)'
                        constraint_status = "VIOLATED"
                        constraint_message = f"Optimization failed and fallback failed. Violations: {', '.join(constraint_violations)}"
            else:
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info("✅ All constraints satisfied in final backtest.")
                print("✅ All constraints satisfied in final backtest.")
                optimized_name = f'{scenario_config["name"]} (Optimized)'
                constraint_status = "SATISFIED"
                constraint_message = "All constraints satisfied"
    
    self.results[optimized_name] = {
        "returns": full_rets if full_rets is not None else pd.Series(dtype=float),
        "display_name": optimized_name,
        "optimal_params": optimal_params,
        "num_trials_for_dsr": actual_num_trials if actual_num_trials is not None else 0,
        "train_end_date": pd.to_datetime(scenario_config.get("train_end_date", "2018-12-31")),
        "best_trial_obj": best_trial_obj,  # Pass the best trial object
        "constraint_status": constraint_status,
        "constraint_message": constraint_message,
        "constraint_violations": constraint_violations if constraint_violations else [],
        "constraints_config": constraints_config
    }
    if self.logger.isEnabledFor(logging.INFO):
        self.logger.info(f"Full backtest with optimized parameters completed for {scenario_config['name']}.")
    
    # Generate comprehensive optimization report
    try:
        self._generate_optimization_report(
            scenario_config=scenario_config,
            optimal_params=optimal_params,
            full_rets=full_rets,
            best_trial_obj=best_trial_obj,
            actual_num_trials=actual_num_trials
        )
    except Exception as e:
        self.logger.error(f"Failed to generate optimization report: {e}")
        self.logger.debug("Report generation error details:", exc_info=True)

def _generate_optimization_report(self, scenario_config, optimal_params, full_rets, best_trial_obj, actual_num_trials):
        """Generate comprehensive optimization report with performance analysis."""
        
        strategy_name = scenario_config["name"]
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(f"Generating comprehensive optimization report for {strategy_name}")
        
        # Calculate performance metrics for the optimized strategy
        if full_rets is not None and not full_rets.empty:
            # Get benchmark data for comparison
            benchmark_ticker = self.global_config["benchmark"]
            
            # Extract benchmark returns from the same period as strategy returns
            if hasattr(self, 'daily_data_ohlc') and self.daily_data_ohlc is not None:
                if isinstance(self.daily_data_ohlc.columns, pd.MultiIndex):
                    benchmark_data = self.daily_data_ohlc.xs(benchmark_ticker, level='Ticker', axis=1)['Close']
                else:
                    benchmark_data = self.daily_data_ohlc[benchmark_ticker]
                
                # Align benchmark data with strategy returns
                benchmark_aligned = benchmark_data.reindex(full_rets.index)
                benchmark_returns = benchmark_aligned.pct_change(fill_method=None).fillna(0)
            else:
                # Fallback: create dummy benchmark returns
                benchmark_returns = pd.Series(0.0, index=full_rets.index)
            
            # Calculate comprehensive performance metrics
            performance_metrics = calculate_metrics(full_rets, benchmark_returns, benchmark_ticker)
        else:
            self.logger.warning("No returns data available for performance metrics calculation")
            performance_metrics = {}
        
        # Prepare optimization results data
        optimization_results = {
            "strategy_name": strategy_name,
            "optimal_parameters": optimal_params,
            "performance_metrics": performance_metrics,
            "optimization_metadata": {
                "num_trials": actual_num_trials,
                "optimizer_type": getattr(self.args, "optimizer", "optuna"),
                "optimization_date": pd.Timestamp.now().isoformat(),
                "global_config": self.global_config
            }
        }
        
        # Add trial data if available from best_trial_obj
        if best_trial_obj and hasattr(best_trial_obj, 'study'):
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
                    
                    # Add user attributes if available
                    if trial.user_attrs:
                        trial_data["user_attrs"] = trial.user_attrs
                    
                    trials_data.append(trial_data)
                
                optimization_results["trials_data"] = trials_data
                optimization_results["best_trial_number"] = best_trial_obj.number
                
                # Calculate parameter importance if we have enough trials
                if len(trials_data) > 10:
                    try:
                        import optuna
                        param_importance = optuna.importance.get_param_importances(study)
                        optimization_results["parameter_importance"] = param_importance
                    except Exception as e:
                        self.logger.warning(f"Could not calculate parameter importance: {e}")
                
            except Exception as e:
                self.logger.warning(f"Could not extract trial data from study: {e}")
        
        # Get constraint information from results
        optimized_result_name = list(self.results.keys())[-1]  # Get the most recent result
        result_data = self.results[optimized_result_name]
        
        # Prepare additional information
        additional_info = {
            "num_trials": actual_num_trials,
            "best_trial_number": best_trial_obj.number,
            "optimization_time": "Not tracked",  # Could be enhanced to track actual time
            "random_seed": getattr(self, 'random_state', None),
            "constraint_info": {
                "status": result_data.get("constraint_status", "UNKNOWN"),
                "message": result_data.get("constraint_message", ""),
                "violations": result_data.get("constraint_violations", []),
                "constraints_config": result_data.get("constraints_config", [])
            }
        }
        
        # Generate the comprehensive report
        try:
            report_path = create_optimization_report(
                strategy_name=strategy_name,
                optimization_results=optimization_results,
                performance_metrics=performance_metrics,
                optimal_parameters=optimal_params,
                plots_source_dir="plots",
                run_id=None,  # Will auto-generate based on timestamp
                additional_info=additional_info
            )
            
            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(f"Comprehensive optimization report generated: {report_path}")
            print(f"\nOptimization Report Generated: {report_path}")
            print(f"Report directory contains:")
            print(f"   - optimization_report.md (Main report)")
            print(f"   - plots/ (All generated visualizations)")
            print(f"   - data/ (Raw optimization data)")
            
        except Exception as e:
            self.logger.error(f"Failed to create optimization report: {e}")
            raise