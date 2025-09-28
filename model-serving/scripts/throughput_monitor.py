#!/usr/bin/env uv run python
"""
Throughput monitoring and alerting system.

This script continuously monitors throughput performance and alerts when
it falls below acceptable thresholds.
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import threading
import queue

# Add model-serving to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarking import BenchmarkEngine, BenchmarkConfig


@dataclass
class ThroughputAlert:
    """Throughput alert data structure."""
    timestamp: str
    current_throughput: float
    threshold: float
    severity: str  # "warning", "critical"
    message: str


class ThroughputMonitor:
    """Continuous throughput monitoring with alerting."""
    
    def __init__(self, 
                 threshold_rps: float = 200.0,
                 check_interval: int = 30,
                 alert_window: int = 300,  # 5 minutes
                 warning_threshold: float = 0.8,  # 80% of threshold
                 critical_threshold: float = 0.6):  # 60% of threshold
        
        self.threshold_rps = threshold_rps
        self.check_interval = check_interval
        self.alert_window = alert_window
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
        self.alerts: List[ThroughputAlert] = []
        self.throughput_history: List[Dict[str, Any]] = []
        self.monitoring = False
        self.alert_queue = queue.Queue()
        
        # Create benchmark configuration for monitoring
        self.config = BenchmarkConfig(
            warmup_iterations=2,
            measurement_iterations=10,
            throughput_threshold_rps=threshold_rps
        )
        self.engine = BenchmarkEngine(self.config)
    
    def create_monitoring_inference_function(self):
        """Create inference function optimized for monitoring."""
        import random
        
        def mock_inference(observations, deterministic=False):
            """Mock inference function for monitoring."""
            # Simulate realistic inference time
            time.sleep(0.002)  # 2ms base delay
            
            # Return mock actions
            actions = []
            for obs in observations:
                action = {
                    "throttle": random.uniform(0.0, 1.0),
                    "brake": random.uniform(0.0, 1.0),
                    "steer": random.uniform(-1.0, 1.0)
                }
                actions.append(action)
            
            return actions
        
        return mock_inference
    
    def measure_current_throughput(self) -> Dict[str, Any]:
        """Measure current throughput performance."""
        try:
            inference_func = self.create_monitoring_inference_function()
            
            # Quick throughput measurement
            result = asyncio.run(self.engine.run_benchmark(inference_func, batch_size=1))
            
            measurement = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "throughput_rps": result.throughput_stats.requests_per_second,
                "p50_latency_ms": result.latency_stats.p50_ms,
                "p95_latency_ms": result.latency_stats.p95_ms,
                "memory_usage_mb": result.memory_stats.peak_memory_mb,
                "success": result.overall_success
            }
            
            return measurement
            
        except Exception as e:
            print(f"Error measuring throughput: {e}")
            return {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "throughput_rps": 0.0,
                "error": str(e),
                "success": False
            }
    
    def check_throughput_threshold(self, measurement: Dict[str, Any]) -> Optional[ThroughputAlert]:
        """Check if throughput measurement triggers an alert."""
        if not measurement.get("success", False):
            return ThroughputAlert(
                timestamp=measurement["timestamp"],
                current_throughput=0.0,
                threshold=self.threshold_rps,
                severity="critical",
                message="Throughput measurement failed"
            )
        
        current_throughput = measurement["throughput_rps"]
        threshold_ratio = current_throughput / self.threshold_rps
        
        if threshold_ratio <= self.critical_threshold:
            return ThroughputAlert(
                timestamp=measurement["timestamp"],
                current_throughput=current_throughput,
                threshold=self.threshold_rps,
                severity="critical",
                message=f"Critical throughput drop: {current_throughput:.1f} RPS (expected: {self.threshold_rps} RPS)"
            )
        elif threshold_ratio <= self.warning_threshold:
            return ThroughputAlert(
                timestamp=measurement["timestamp"],
                current_throughput=current_throughput,
                threshold=self.threshold_rps,
                severity="warning",
                message=f"Warning: Throughput below warning threshold: {current_throughput:.1f} RPS (expected: {self.threshold_rps} RPS)"
            )
        
        return None
    
    def cleanup_old_data(self):
        """Remove old measurements and alerts outside the alert window."""
        current_time = time.time()
        cutoff_time = current_time - self.alert_window
        
        # Clean up throughput history
        self.throughput_history = [
            m for m in self.throughput_history
            if time.mktime(time.strptime(m["timestamp"], "%Y-%m-%d %H:%M:%S UTC")) > cutoff_time
        ]
        
        # Clean up alerts
        self.alerts = [
            a for a in self.alerts
            if time.mktime(time.strptime(a.timestamp, "%Y-%m-%d %H:%M:%S UTC")) > cutoff_time
        ]
    
    def monitor_loop(self):
        """Main monitoring loop."""
        print(f"Starting throughput monitoring...")
        print(f"Threshold: {self.threshold_rps} RPS")
        print(f"Check interval: {self.check_interval} seconds")
        print(f"Warning threshold: {self.warning_threshold * 100:.0f}%")
        print(f"Critical threshold: {self.critical_threshold * 100:.0f}%")
        print("-" * 50)
        
        while self.monitoring:
            try:
                # Measure current throughput
                measurement = self.measure_current_throughput()
                self.throughput_history.append(measurement)
                
                # Check for alerts
                alert = self.check_throughput_threshold(measurement)
                if alert:
                    self.alerts.append(alert)
                    self.alert_queue.put(alert)
                    self.print_alert(alert)
                
                # Print status
                self.print_status(measurement)
                
                # Cleanup old data
                self.cleanup_old_data()
                
                # Wait for next check
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def print_alert(self, alert: ThroughputAlert):
        """Print throughput alert."""
        severity_icon = "🚨" if alert.severity == "critical" else "⚠️"
        print(f"\n{severity_icon} THROUGHPUT ALERT - {alert.severity.upper()}")
        print(f"Time: {alert.timestamp}")
        print(f"Current: {alert.current_throughput:.1f} RPS")
        print(f"Threshold: {alert.threshold} RPS")
        print(f"Message: {alert.message}")
        print("-" * 50)
    
    def print_status(self, measurement: Dict[str, Any]):
        """Print current status."""
        throughput = measurement["throughput_rps"]
        p50_latency = measurement.get("p50_latency_ms", 0)
        memory = measurement.get("memory_usage_mb", 0)
        
        # Calculate efficiency
        efficiency = (throughput / self.threshold_rps) * 100
        
        status_icon = "✅" if efficiency >= 100 else "⚠️" if efficiency >= 80 else "❌"
        
        print(f"{status_icon} {measurement['timestamp']} | "
              f"Throughput: {throughput:.1f} RPS ({efficiency:.1f}%) | "
              f"P50: {p50_latency:.1f}ms | "
              f"Memory: {memory:.1f}MB")
    
    def start_monitoring(self):
        """Start throughput monitoring."""
        self.monitoring = True
        self.monitor_loop()
    
    def stop_monitoring(self):
        """Stop throughput monitoring."""
        self.monitoring = False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        if not self.throughput_history:
            return {"error": "No measurements available"}
        
        recent_measurements = self.throughput_history[-10:]  # Last 10 measurements
        throughputs = [m["throughput_rps"] for m in recent_measurements if m.get("success", False)]
        
        if not throughputs:
            return {"error": "No successful measurements"}
        
        avg_throughput = sum(throughputs) / len(throughputs)
        min_throughput = min(throughputs)
        max_throughput = max(throughputs)
        
        return {
            "measurements_count": len(self.throughput_history),
            "recent_measurements": len(recent_measurements),
            "average_throughput_rps": avg_throughput,
            "min_throughput_rps": min_throughput,
            "max_throughput_rps": max_throughput,
            "threshold_rps": self.threshold_rps,
            "efficiency_percent": (avg_throughput / self.threshold_rps) * 100,
            "alerts_count": len(self.alerts),
            "critical_alerts": len([a for a in self.alerts if a.severity == "critical"]),
            "warning_alerts": len([a for a in self.alerts if a.severity == "warning"])
        }


def main():
    """Main entry point for throughput monitoring."""
    parser = argparse.ArgumentParser(
        description="Monitor throughput performance with alerting"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=200.0,
        help="Throughput threshold in RPS (default: 200.0)"
    )
    
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=30,
        help="Check interval in seconds (default: 30)"
    )
    
    parser.add_argument(
        "--duration", "-d",
        type=int,
        help="Monitoring duration in seconds (default: run indefinitely)"
    )
    
    parser.add_argument(
        "--warning-threshold",
        type=float,
        default=0.8,
        help="Warning threshold as fraction of main threshold (default: 0.8)"
    )
    
    parser.add_argument(
        "--critical-threshold",
        type=float,
        default=0.6,
        help="Critical threshold as fraction of main threshold (default: 0.6)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for monitoring summary (JSON format)"
    )
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = ThroughputMonitor(
        threshold_rps=args.threshold,
        check_interval=args.interval,
        warning_threshold=args.warning_threshold,
        critical_threshold=args.critical_threshold
    )
    
    try:
        if args.duration:
            # Run for specified duration
            print(f"Running monitoring for {args.duration} seconds...")
            monitor.start_monitoring()
            time.sleep(args.duration)
            monitor.stop_monitoring()
        else:
            # Run indefinitely
            monitor.start_monitoring()
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
        monitor.stop_monitoring()
    
    finally:
        # Generate summary
        summary = monitor.get_summary()
        
        print("\n" + "="*50)
        print("MONITORING SUMMARY")
        print("="*50)
        
        if "error" in summary:
            print(f"Error: {summary['error']}")
        else:
            print(f"Measurements: {summary['measurements_count']}")
            print(f"Average Throughput: {summary['average_throughput_rps']:.1f} RPS")
            print(f"Min Throughput: {summary['min_throughput_rps']:.1f} RPS")
            print(f"Max Throughput: {summary['max_throughput_rps']:.1f} RPS")
            print(f"Efficiency: {summary['efficiency_percent']:.1f}%")
            print(f"Alerts: {summary['alerts_count']} (Critical: {summary['critical_alerts']}, Warning: {summary['warning_alerts']})")
        
        # Save summary if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary saved to: {args.output}")


if __name__ == "__main__":
    import asyncio
    main()
