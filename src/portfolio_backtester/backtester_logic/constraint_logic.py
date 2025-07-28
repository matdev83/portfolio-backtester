import logging
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)

def display_constraint_violation_warning(results: dict):
    """Display a prominent constraint violation warning."""
    console = Console()
    
    warning_text = Text()
    warning_text.append("🚨 CONSTRAINT VIOLATION DETECTED! 🚨\n\n", style="bold red")
    
    for name, result_data in results.items():
        constraint_status = result_data.get("constraint_status", "UNKNOWN")
        if constraint_status == "VIOLATED":
            warning_text.append(f"Strategy: {result_data.get('display_name', name)}\n", style="bold yellow")
            warning_text.append(f"Status: {constraint_status}\n", style="red")
            warning_text.append(f"Issue: {result_data.get('constraint_message', 'Unknown constraint violation')}\n", style="red")
            
            violations = result_data.get("constraint_violations", [])
            if violations:
                warning_text.append("Violations:\n", style="bold red")
                for i, violation in enumerate(violations, 1):
                    warning_text.append(f"  {i}. {violation}\n", style="red")
            
            warning_text.append("\n")
    
    warning_text.append("💡 RECOMMENDATIONS:\n", style="bold cyan")
    warning_text.append("• Relax the constraint limits (e.g., increase max volatility)\n", style="cyan")
    warning_text.append("• Modify strategy parameters to reduce risk\n", style="cyan")
    warning_text.append("• Consider different optimization targets\n", style="cyan")
    warning_text.append("• Review strategy configuration and universe\n", style="cyan")
    
    panel = Panel(
        warning_text,
        title="⚠️  OPTIMIZATION CONSTRAINT FAILURE ⚠️",
        title_align="center",
        border_style="red",
        expand=False
    )
    
    console.print(panel)
    console.print()  # Add spacing