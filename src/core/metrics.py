"""
Performance monitoring and metrics collection for the voice transformation pipeline.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
import statistics
import threading


class MetricType(str, Enum):
    """Types of metrics that can be collected."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    QUEUE_SIZE = "queue_size"
    RESOURCE_USAGE = "resource_usage"


@dataclass
class MetricSample:
    """A single metric measurement."""
    timestamp: datetime
    value: float
    stage: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricStats:
    """Statistical summary of metric samples."""
    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    min_value: float = float('inf')
    max_value: float = float('-inf')
    std_dev: float = 0.0
    percentile_95: float = 0.0
    percentile_99: float = 0.0
    
    def update(self, values: List[float]):
        """Update statistics from a list of values."""
        if not values:
            return
        
        self.count = len(values)
        self.mean = statistics.mean(values)
        self.median = statistics.median(values)
        self.min_value = min(values)
        self.max_value = max(values)
        
        if len(values) > 1:
            self.std_dev = statistics.stdev(values)
        
        sorted_values = sorted(values)
        if len(sorted_values) >= 20:  # Only calculate percentiles for sufficient data
            # Use proper percentile calculation (0-based indexing)
            p95_index = max(0, int(0.95 * len(sorted_values)) - 1)
            p99_index = max(0, int(0.99 * len(sorted_values)) - 1)
            self.percentile_95 = sorted_values[p95_index]
            self.percentile_99 = sorted_values[p99_index]


class MetricCollector:
    """Collects and manages metrics for a specific metric type."""
    
    def __init__(self, metric_type: MetricType, max_samples: int = 1000):
        self.metric_type = metric_type
        self.max_samples = max_samples
        self.samples: deque = deque(maxlen=max_samples)
        self._lock = threading.Lock()
    
    def add_sample(self, value: float, stage: Optional[str] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a new metric sample."""
        with self._lock:
            sample = MetricSample(
                timestamp=datetime.now(),
                value=value,
                stage=stage,
                metadata=metadata or {}
            )
            self.samples.append(sample)
    
    def get_recent_samples(self, duration: timedelta) -> List[MetricSample]:
        """Get samples from the last specified duration."""
        cutoff_time = datetime.now() - duration
        with self._lock:
            return [s for s in self.samples if s.timestamp >= cutoff_time]
    
    def get_stats(self, duration: Optional[timedelta] = None) -> MetricStats:
        """Get statistical summary of samples."""
        if duration:
            samples = self.get_recent_samples(duration)
        else:
            with self._lock:
                samples = list(self.samples)
        
        stats = MetricStats()
        if samples:
            values = [s.value for s in samples]
            stats.update(values)
        
        return stats
    
    def get_stage_stats(self, stage: str, duration: Optional[timedelta] = None) -> MetricStats:
        """Get statistics for a specific pipeline stage."""
        if duration:
            samples = self.get_recent_samples(duration)
        else:
            with self._lock:
                samples = list(self.samples)
        
        stage_samples = [s for s in samples if s.stage == stage]
        stats = MetricStats()
        if stage_samples:
            values = [s.value for s in stage_samples]
            stats.update(values)
        
        return stats
    
    def clear(self) -> None:
        """Clear all samples."""
        with self._lock:
            self.samples.clear()


class PerformanceMonitor:
    """Main performance monitoring system for the pipeline."""
    
    def __init__(self, max_samples_per_metric: int = 1000):
        self.max_samples = max_samples_per_metric
        self.collectors: Dict[MetricType, MetricCollector] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize collectors for each metric type
        for metric_type in MetricType:
            self.collectors[metric_type] = MetricCollector(metric_type, max_samples_per_metric)
        
        # Performance thresholds for alerts
        self.thresholds = {
            MetricType.LATENCY: {
                'warning': 1000.0,  # 1 second
                'critical': 2000.0  # 2 seconds
            },
            MetricType.ERROR_RATE: {
                'warning': 0.05,    # 5%
                'critical': 0.10    # 10%
            },
            MetricType.QUEUE_SIZE: {
                'warning': 8,       # 80% of max queue size (10)
                'critical': 10      # 100% of max queue size
            }
        }
        
        # Alert tracking
        self._last_alerts: Dict[str, datetime] = {}
        self._alert_cooldown = timedelta(minutes=5)
    
    def record_latency(self, stage: str, latency_ms: float, 
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record latency measurement for a pipeline stage."""
        self.collectors[MetricType.LATENCY].add_sample(
            latency_ms, stage, metadata
        )
        
        # Check for latency alerts
        self._check_latency_alert(stage, latency_ms)
    
    def record_throughput(self, stage: str, items_per_second: float,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record throughput measurement for a pipeline stage."""
        self.collectors[MetricType.THROUGHPUT].add_sample(
            items_per_second, stage, metadata
        )
    
    def record_error(self, stage: str, error_count: int = 1,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record error occurrence for a pipeline stage."""
        self.collectors[MetricType.ERROR_RATE].add_sample(
            error_count, stage, metadata
        )
    
    def record_queue_size(self, queue_name: str, size: int,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record queue size measurement."""
        self.collectors[MetricType.QUEUE_SIZE].add_sample(
            size, queue_name, metadata
        )
        
        # Check for queue size alerts
        self._check_queue_alert(queue_name, size)
    
    def record_resource_usage(self, resource_type: str, usage_percent: float,
                             metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record resource usage measurement (CPU, GPU, memory)."""
        self.collectors[MetricType.RESOURCE_USAGE].add_sample(
            usage_percent, resource_type, metadata
        )
    
    def get_performance_summary(self, duration: timedelta = timedelta(minutes=5)) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'duration_minutes': duration.total_seconds() / 60,
            'metrics': {}
        }
        
        for metric_type, collector in self.collectors.items():
            stats = collector.get_stats(duration)
            summary['metrics'][metric_type.value] = {
                'count': stats.count,
                'mean': stats.mean,
                'median': stats.median,
                'min': stats.min_value if stats.min_value != float('inf') else 0,
                'max': stats.max_value if stats.max_value != float('-inf') else 0,
                'std_dev': stats.std_dev,
                'p95': stats.percentile_95,
                'p99': stats.percentile_99
            }
        
        return summary
    
    def get_stage_performance(self, stage: str, 
                            duration: timedelta = timedelta(minutes=5)) -> Dict[str, Any]:
        """Get performance metrics for a specific pipeline stage."""
        stage_metrics = {}
        
        for metric_type, collector in self.collectors.items():
            stats = collector.get_stage_stats(stage, duration)
            if stats.count > 0:
                stage_metrics[metric_type.value] = {
                    'count': stats.count,
                    'mean': stats.mean,
                    'median': stats.median,
                    'min': stats.min_value,
                    'max': stats.max_value,
                    'std_dev': stats.std_dev,
                    'p95': stats.percentile_95,
                    'p99': stats.percentile_99
                }
        
        return {
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'duration_minutes': duration.total_seconds() / 60,
            'metrics': stage_metrics
        }
    
    def get_optimization_suggestions(self) -> List[str]:
        """Generate optimization suggestions based on current metrics."""
        suggestions = []
        recent_duration = timedelta(minutes=5)
        
        # Check latency issues
        latency_stats = self.collectors[MetricType.LATENCY].get_stats(recent_duration)
        if latency_stats.count > 0:
            if latency_stats.mean > self.thresholds[MetricType.LATENCY]['critical']:
                suggestions.append(
                    f"High average latency detected ({latency_stats.mean:.1f}ms). "
                    "Consider reducing model sizes or enabling GPU acceleration."
                )
            elif latency_stats.percentile_95 > self.thresholds[MetricType.LATENCY]['warning']:
                suggestions.append(
                    f"High P95 latency detected ({latency_stats.percentile_95:.1f}ms). "
                    "Consider optimizing the slowest pipeline stages."
                )
        
        # Check queue sizes
        queue_stats = self.collectors[MetricType.QUEUE_SIZE].get_stats(recent_duration)
        if queue_stats.count > 0 and queue_stats.mean > self.thresholds[MetricType.QUEUE_SIZE]['warning']:
            suggestions.append(
                f"High average queue size ({queue_stats.mean:.1f}). "
                "Consider increasing processing capacity or reducing input rate."
            )
        
        # Check error rates
        error_stats = self.collectors[MetricType.ERROR_RATE].get_stats(recent_duration)
        if error_stats.count > 0:
            error_rate = error_stats.mean / max(1, latency_stats.count)  # Approximate error rate
            if error_rate > self.thresholds[MetricType.ERROR_RATE]['warning']:
                suggestions.append(
                    f"High error rate detected ({error_rate:.2%}). "
                    "Check logs for recurring issues and consider implementing retry logic."
                )
        
        # Check resource usage
        resource_stats = self.collectors[MetricType.RESOURCE_USAGE].get_stats(recent_duration)
        if resource_stats.count > 0:
            if resource_stats.mean > 90:
                suggestions.append(
                    f"High resource usage ({resource_stats.mean:.1f}%). "
                    "Consider reducing batch sizes or model complexity."
                )
            elif resource_stats.mean < 30:
                suggestions.append(
                    f"Low resource usage ({resource_stats.mean:.1f}%). "
                    "Consider increasing batch sizes or model complexity for better performance."
                )
        
        return suggestions
    
    def _check_latency_alert(self, stage: str, latency_ms: float) -> None:
        """Check if latency alert should be triggered."""
        alert_key = f"latency_{stage}"
        
        if self._should_send_alert(alert_key):
            if latency_ms > self.thresholds[MetricType.LATENCY]['critical']:
                self.logger.warning(
                    f"CRITICAL: High latency in {stage}: {latency_ms:.1f}ms "
                    f"(threshold: {self.thresholds[MetricType.LATENCY]['critical']}ms)"
                )
                self._last_alerts[alert_key] = datetime.now()
            elif latency_ms > self.thresholds[MetricType.LATENCY]['warning']:
                self.logger.warning(
                    f"WARNING: Elevated latency in {stage}: {latency_ms:.1f}ms "
                    f"(threshold: {self.thresholds[MetricType.LATENCY]['warning']}ms)"
                )
                self._last_alerts[alert_key] = datetime.now()
    
    def _check_queue_alert(self, queue_name: str, size: int) -> None:
        """Check if queue size alert should be triggered."""
        alert_key = f"queue_{queue_name}"
        
        if self._should_send_alert(alert_key):
            if size >= self.thresholds[MetricType.QUEUE_SIZE]['critical']:
                self.logger.warning(
                    f"CRITICAL: Queue {queue_name} is full: {size} items"
                )
                self._last_alerts[alert_key] = datetime.now()
            elif size >= self.thresholds[MetricType.QUEUE_SIZE]['warning']:
                self.logger.warning(
                    f"WARNING: Queue {queue_name} is nearly full: {size} items"
                )
                self._last_alerts[alert_key] = datetime.now()
    
    def _should_send_alert(self, alert_key: str) -> bool:
        """Check if enough time has passed since last alert."""
        last_alert = self._last_alerts.get(alert_key)
        if last_alert is None:
            return True
        
        return datetime.now() - last_alert > self._alert_cooldown
    
    def reset_metrics(self) -> None:
        """Reset all collected metrics."""
        for collector in self.collectors.values():
            collector.clear()
        self._last_alerts.clear()
        self.logger.info("All metrics have been reset")
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        summary = self.get_performance_summary()
        
        if format.lower() == "json":
            import json
            return json.dumps(summary, indent=2)
        elif format.lower() == "csv":
            # Simple CSV export for basic metrics
            lines = ["metric_type,stage,count,mean,median,min,max,std_dev,p95,p99"]
            for metric_type, data in summary['metrics'].items():
                lines.append(
                    f"{metric_type},,{data['count']},{data['mean']:.2f},"
                    f"{data['median']:.2f},{data['min']:.2f},{data['max']:.2f},"
                    f"{data['std_dev']:.2f},{data['p95']:.2f},{data['p99']:.2f}"
                )
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")


class LatencyTracker:
    """Context manager for tracking operation latency."""
    
    def __init__(self, monitor: PerformanceMonitor, stage: str, 
                 metadata: Optional[Dict[str, Any]] = None):
        self.monitor = monitor
        self.stage = stage
        self.metadata = metadata or {}
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            latency_ms = (time.time() - self.start_time) * 1000
            self.monitor.record_latency(self.stage, latency_ms, self.metadata)
            
            # Record error if exception occurred
            if exc_type is not None:
                self.monitor.record_error(self.stage, 1, {
                    'error_type': exc_type.__name__,
                    'error_message': str(exc_val),
                    **self.metadata
                })