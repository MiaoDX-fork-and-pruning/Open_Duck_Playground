"""Performance monitoring utilities for Newton integration."""

import time
import jax
import jax.numpy as jp
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PerformanceMetrics:
    """Track performance metrics for physics backends."""
    backend: str
    step_times: List[float] = field(default_factory=list)
    total_steps: int = 0
    start_time: Optional[float] = None
    
    def start_timer(self):
        """Start timing a step."""
        self.start_time = time.time()
        
    def end_timer(self):
        """End timing and record step time."""
        if self.start_time is not None:
            step_time = time.time() - self.start_time
            self.step_times.append(step_time)
            self.total_steps += 1
            self.start_time = None
            
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.step_times:
            return {}
            
        step_times_ms = [t * 1000 for t in self.step_times]
        recent_times = step_times_ms[-1000:]  # Last 1000 steps
        
        return {
            f"{self.backend}_mean_step_ms": np.mean(recent_times),
            f"{self.backend}_std_step_ms": np.std(recent_times),
            f"{self.backend}_min_step_ms": np.min(recent_times),
            f"{self.backend}_max_step_ms": np.max(recent_times),
            f"{self.backend}_steps_per_second": 1000.0 / np.mean(recent_times),
            f"{self.backend}_total_steps": self.total_steps
        }
        
    def log_stats(self, every_n_steps: int = 1000):
        """Log statistics every N steps."""
        if self.total_steps % every_n_steps == 0 and self.total_steps > 0:
            stats = self.get_stats()
            print(f"\n[{self.backend}] Performance after {self.total_steps} steps:")
            print(f"  Mean step time: {stats[f'{self.backend}_mean_step_ms']:.2f} ms")
            print(f"  Steps per second: {stats[f'{self.backend}_steps_per_second']:.1f}")
            print(f"  Std dev: {stats[f'{self.backend}_std_step_ms']:.2f} ms")


class PerformanceLogger:
    """Log and compare performance between backends."""
    
    def __init__(self):
        self.metrics: Dict[str, PerformanceMetrics] = {}
        
    def get_metrics(self, backend: str) -> PerformanceMetrics:
        """Get or create metrics for a backend."""
        if backend not in self.metrics:
            self.metrics[backend] = PerformanceMetrics(backend)
        return self.metrics[backend]
        
    def start_step(self, backend: str):
        """Start timing a step."""
        self.get_metrics(backend).start_timer()
        
    def end_step(self, backend: str):
        """End timing a step."""
        metrics = self.get_metrics(backend)
        metrics.end_timer()
        metrics.log_stats()
        
    def compare_backends(self):
        """Compare performance between backends."""
        if len(self.metrics) < 2:
            return
            
        print("\n" + "="*60)
        print("Backend Performance Comparison")
        print("="*60)
        
        for backend, metrics in self.metrics.items():
            stats = metrics.get_stats()
            if stats:
                print(f"\n{backend}:")
                print(f"  Steps: {stats[f'{backend}_total_steps']}")
                print(f"  Mean: {stats[f'{backend}_mean_step_ms']:.2f} ms")
                print(f"  Rate: {stats[f'{backend}_steps_per_second']:.1f} steps/s")
                
        # Calculate speedup
        if "Newton" in self.metrics and "MJX" in self.metrics:
            newton_stats = self.metrics["Newton"].get_stats()
            mjx_stats = self.metrics["MJX"].get_stats()
            
            if newton_stats and mjx_stats:
                speedup = (newton_stats["Newton_steps_per_second"] / 
                          mjx_stats["MJX_steps_per_second"])
                print(f"\nNewton speedup: {speedup:.2f}x")
        
        print("="*60)


# Global performance logger instance
_global_logger = PerformanceLogger()


def get_performance_logger() -> PerformanceLogger:
    """Get the global performance logger."""
    return _global_logger


def log_backend_info(backend: str, device: str):
    """Log backend and device information."""
    print(f"\n{'='*60}")
    print(f"Physics Backend: {backend}")
    print(f"Device: {device}")
    print(f"JAX devices: {jax.devices()}")
    print(f"{'='*60}\n")