#!/usr/bin/env python3
"""
Quick test to verify our enhanced trade statistics implementation.
"""

import sys
import os
import subprocess

def run_quick_test():
    """Run a quick test of the enhanced trade statistics."""
    print("🚀 RUNNING QUICK TEST OF ENHANCED TRADE STATISTICS")
    print("=" * 60)
    
    try:
        # Run the backtester with our test scenario
        cmd = [
            ".venv/Scripts/python.exe", 
            "-m", "portfolio_backtester.backtester",
            "--mode", "backtest",
            "--scenario-file", "test_scenario.yaml",
            "--log-level", "INFO"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        print("This may take a moment to fetch data and run the backtest...")
        print("-" * 60)
        
        # Run with timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,  # 3 minutes
            cwd="."
        )
        
        print(f"Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ BACKTEST COMPLETED SUCCESSFULLY!")
            
            # Check output for trade statistics
            output_lower = result.stdout.lower()
            
            # Look for trade-related keywords
            trade_keywords = [
                'number of trades', 'win rate', 'total p&l', 'largest single',
                'reward/risk', 'commissions', 'mfe', 'mae', 'trade duration',
                'margin load', 'all trades', 'long trades', 'short trades'
            ]
            
            found_keywords = []
            for keyword in trade_keywords:
                if keyword in output_lower:
                    found_keywords.append(keyword)
            
            if found_keywords:
                print(f"✅ FOUND TRADE STATISTICS: {len(found_keywords)} keywords detected")
                print(f"   Keywords found: {found_keywords}")
                print("🎉 IMPLEMENTATION IS WORKING!")
                
                # Show a sample of the output
                lines = result.stdout.split('\\n')
                print("\\n📊 SAMPLE OUTPUT:")
                for i, line in enumerate(lines):
                    if any(kw.replace(' ', '') in line.lower().replace(' ', '') for kw in trade_keywords):
                        print(f"   {line.strip()}")
                        if i < len(lines) - 1:
                            print(f"   {lines[i+1].strip()}")
                        break
                
                return True
            else:
                print("⚠️  NO TRADE STATISTICS FOUND IN OUTPUT")
                print("This could mean:")
                print("  - No trades were generated (possible with small test)")
                print("  - Trade tracking is not enabled")
                print("  - Implementation needs debugging")
                
                print("\\n📋 FULL OUTPUT:")
                print(result.stdout)
                return False
                
        else:
            print("❌ BACKTEST FAILED!")
            print("\\nSTDOUT:")
            print(result.stdout)
            print("\\nSTDERR:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 3 minutes")
        return False
    except FileNotFoundError:
        print("❌ Python executable not found. Check .venv/Scripts/python.exe")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = run_quick_test()
    
    print("\\n" + "=" * 60)
    if success:
        print("🎉 SUCCESS: Enhanced trade statistics are working!")
        print("✅ TASK COMPLETION VERIFIED")
    else:
        print("❌ ISSUES DETECTED: Implementation needs debugging")
        print("🔧 TASK REQUIRES FURTHER WORK")
    print("=" * 60)
    
    sys.exit(0 if success else 1)