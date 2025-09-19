"""
Unit tests for the performance monitoring and metrics system.
"""

import pytest
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.core.metrics import (
    MetricType, MetricSample, MetricStats, MetricCollector,
    PerformanceMonitor, LatencyTracker
)


class TestMetricSample:
    """Test metric sample functionality."""
    
    def test_metric_sample_creation(self):
        """Test metric sample is created correctly."""
        timestamp = datetime.now()
        sample = MetricSample(
            timestamp=timestamp,
            value=100.5,
            stage="test_stage",
            metadata={"key": "value"}
        )
        
        assert sample.timestamp == timestamp
        assert sample.value == 100.5
        assert sample.stage == "test_stage"
        assert sample.metadata == {"key": "value"}
    
    def test_metric_sample_defaults(self):
        """Test metric sample default values."""
        timestamp = datetime.now()
        sample = MetricSample(timestamp=timestamp, value=50.0)
        
        assert sample.timestamp == timestamp
        assert sample.value == 50.0
        assert sample.stage is None
        assert sample.metadata == {}


class TestMetricStats:
    """Test metric statistics functionality."""
    
    def test_stats_initialization(self):
        """Test stats are initialized correctly."""
        stats = MetricStats()
        
        assert stats.count == 0
        assert stats.mean == 0.0
        assert stats.median == 0.0
        assert stats.min_value == float('inf')
        assert stats.max_value == float('-inf')
        assert stats.std_dev == 0.0
        assert stats.percentile_95 == 0.0
        assert stats.percentile_99 == 0.0
    
    def test_stats_update_single_value(self):
        """Test stats update with single value."""
        stats = MetricStats()
        stats.update([42.0])
        
        assert stats.count == 1
        assert stats.mean == 42.0
        assert stats.median == 42.0
        assert stats.min_value == 42.0
        assert stats.max_value == 42.0
        assert stats.std_dev == 0.0
    
    def test_stats_update_multiple_values(self):
        """Test stats update with multiple values."""
        stats = MetricStats()
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats.update(values)
        
        assert stats.count == 5
        assert stats.mean == 30.0
        assert stats.median == 30.0
        assert stats.min_value == 10.0
        assert stats.max_value == 50.0
        assert stats.std_dev > 0
    
    def test_stats_update_large_dataset(self):
        """Test stats update with large dataset for percentiles."""
        stats = MetricStats()
        values = list(range(1, 101))  # 1 to 100
        stats.update(values)
        
        assert stats.count == 100
        assert stats.mean == 50.5
        assert stats.median == 50.5
        assert stats.min_value == 1
        assert stats.max_value == 100
        assert stats.percentile_95 == 95
        assert stats.percentile_99 == 99
    
    def test_stats_update_empty_values(self):
        """Test stats update with empty values."""
        stats = MetricStats()
        stats.update([])
        
        assert stats.count == 0
        assert stats.mean == 0.0


class TestMetricCollector:
    """Test metric collector functionality."""
    
    def test_collector_initialization(self):
        """Test collector is initialized correctly."""
        collector = MetricCollector(MetricType.LATENCY, max_samples=100)
        
        assert collector.metric_type == MetricType.LATENCY
        assert collector.max_samples == 100
        assert len(collector.samples) == 0
    
    def test_add_sample(self):
        """Test adding samples to collector."""
        collector = MetricCollector(MetricType.LATENCY)
        
        collector.add_sample(100.5, "test_stage", {"key": "value"})
        
        assert len(collector.samples) == 1
        sample = collector.samples[0]
        assert sample.value == 100.5
        assert sample.stage == "test_stage"
        assert sample.metadata == {"key": "value"}
        assert isinstance(sample.timestamp, datetime)
    
    def test_max_samples_limit(self):
        """Test collector respects max samples limit."""
        collector = MetricCollector(MetricType.LATENCY, max_samples=3)
        
        # Add more samples than the limit
        for i in range(5):
            collector.add_sample(float(i))
        
        # Should only keep the last 3 samples
        assert len(collector.samples) == 3
        values = [s.value for s in collector.samples]
        assert values == [2.0, 3.0, 4.0]
    
    def test_get_recent_samples(self):
        """Test getting recent samples within duration."""
        collector = MetricCollector(MetricType.LATENCY)
        
        # Add samples with different timestamps
        now = datetime.now()
        with patch('src.core.metrics.datetime') as mock_datetime:
            # Add old sample
            mock_datetime.now.return_value = now - timedelta(minutes=10)
            collector.add_sample(10.0)
            
            # Add recent sample
            mock_datetime.now.return_value = now
            collector.add_sample(20.0)
        
        # Get samples from last 5 minutes
        recent_samples = collector.get_recent_samples(timedelta(minutes=5))
        
        assert len(recent_samples) == 1
        assert recent_samples[0].value == 20.0
    
    def test_get_stats(self):
        """Test getting statistics from collector."""
        collector = MetricCollector(MetricType.LATENCY)
        
        # Add some samples
        values = [10.0, 20.0, 30.0]
        for value in values:
            collector.add_sample(value)
        
        stats = collector.get_stats()
        
        assert stats.count == 3
        assert stats.mean == 20.0
        assert stats.min_value == 10.0
        assert stats.max_value == 30.0
    
    def test_get_stage_stats(self):
        """Test getting statistics for specific stage."""
        collector = MetricCollector(MetricType.LATENCY)
        
        # Add samples for different stages
        collector.add_sample(10.0, "stage1")
        collector.add_sample(20.0, "stage2")
        collector.add_sample(30.0, "stage1")
        
        stage1_stats = collector.get_stage_stats("stage1")
        
        assert stage1_stats.count == 2
        assert stage1_stats.mean == 20.0  # (10 + 30) / 2
    
    def test_clear(self):
        """Test clearing collector samples."""
        collector = MetricCollector(MetricType.LATENCY)
        
        collector.add_sample(10.0)
        collector.add_sample(20.0)
        assert len(collector.samples) == 2
        
        collector.clear()
        assert len(collector.samples) == 0


class TestPerformanceMonitor:
    """Test performance monitor functionality."""
    
    def test_monitor_initialization(self):
        """Test monitor is initialized correctly."""
        monitor = PerformanceMonitor(max_samples_per_metric=500)
        
        assert monitor.max_samples == 500
        assert len(monitor.collectors) == len(MetricType)
        
        for metric_type in MetricType:
            assert metric_type in monitor.collectors
            assert monitor.collectors[metric_type].metric_type == metric_type
    
    def test_record_latency(self):
        """Test recording latency measurements."""
        monitor = PerformanceMonitor()
        
        monitor.record_latency("test_stage", 100.5, {"key": "value"})
        
        collector = monitor.collectors[MetricType.LATENCY]
        assert len(collector.samples) == 1
        
        sample = collector.samples[0]
        assert sample.value == 100.5
        assert sample.stage == "test_stage"
        assert sample.metadata == {"key": "value"}
    
    def test_record_throughput(self):
        """Test recording throughput measurements."""
        monitor = PerformanceMonitor()
        
        monitor.record_throughput("test_stage", 50.0)
        
        collector = monitor.collectors[MetricType.THROUGHPUT]
        assert len(collector.samples) == 1
        assert collector.samples[0].value == 50.0
        assert collector.samples[0].stage == "test_stage"
    
    def test_record_error(self):
        """Test recording error measurements."""
        monitor = PerformanceMonitor()
        
        monitor.record_error("test_stage", 2, {"error_type": "TestError"})
        
        collector = monitor.collectors[MetricType.ERROR_RATE]
        assert len(collector.samples) == 1
        assert collector.samples[0].value == 2
        assert collector.samples[0].stage == "test_stage"
        assert collector.samples[0].metadata == {"error_type": "TestError"}
    
    def test_record_queue_size(self):
        """Test recording queue size measurements."""
        monitor = PerformanceMonitor()
        
        monitor.record_queue_size("test_queue", 5)
        
        collector = monitor.collectors[MetricType.QUEUE_SIZE]
        assert len(collector.samples) == 1
        assert collector.samples[0].value == 5
        assert collector.samples[0].stage == "test_queue"
    
    def test_record_resource_usage(self):
        """Test recording resource usage measurements."""
        monitor = PerformanceMonitor()
        
        monitor.record_resource_usage("gpu", 75.5)
        
        collector = monitor.collectors[MetricType.RESOURCE_USAGE]
        assert len(collector.samples) == 1
        assert collector.samples[0].value == 75.5
        assert collector.samples[0].stage == "gpu"
    
    def test_get_performance_summary(self):
        """Test getting performance summary."""
        monitor = PerformanceMonitor()
        
        # Add some sample data
        monitor.record_latency("test_stage", 100.0)
        monitor.record_latency("test_stage", 200.0)
        monitor.record_throughput("test_stage", 50.0)
        
        summary = monitor.get_performance_summary()
        
        assert "timestamp" in summary
        assert "duration_minutes" in summary
        assert "metrics" in summary
        
        # Check latency metrics
        latency_metrics = summary["metrics"]["latency"]
        assert latency_metrics["count"] == 2
        assert latency_metrics["mean"] == 150.0
        assert latency_metrics["min"] == 100.0
        assert latency_metrics["max"] == 200.0
        
        # Check throughput metrics
        throughput_metrics = summary["metrics"]["throughput"]
        assert throughput_metrics["count"] == 1
        assert throughput_metrics["mean"] == 50.0
    
    def test_get_stage_performance(self):
        """Test getting performance for specific stage."""
        monitor = PerformanceMonitor()
        
        # Add data for different stages
        monitor.record_latency("stage1", 100.0)
        monitor.record_latency("stage2", 200.0)
        monitor.record_latency("stage1", 150.0)
        
        stage1_performance = monitor.get_stage_performance("stage1")
        
        assert stage1_performance["stage"] == "stage1"
        assert "timestamp" in stage1_performance
        assert "metrics" in stage1_performance
        
        latency_metrics = stage1_performance["metrics"]["latency"]
        assert latency_metrics["count"] == 2
        assert latency_metrics["mean"] == 125.0  # (100 + 150) / 2
    
    def test_get_optimization_suggestions(self):
        """Test getting optimization suggestions."""
        monitor = PerformanceMonitor()
        
        # Add high latency data to trigger suggestions
        for _ in range(10):
            monitor.record_latency("test_stage", 2500.0)  # Above critical threshold
        
        suggestions = monitor.get_optimization_suggestions()
        
        assert len(suggestions) > 0
        assert any("High average latency" in suggestion for suggestion in suggestions)
    
    def test_latency_alert_critical(self):
        """Test critical latency alert."""
        monitor = PerformanceMonitor()
        
        with patch.object(monitor.logger, 'warning') as mock_warning:
            monitor.record_latency("test_stage", 2500.0)  # Above critical threshold
            
            mock_warning.assert_called_once()
            assert "CRITICAL" in mock_warning.call_args[0][0]
            assert "High latency" in mock_warning.call_args[0][0]
    
    def test_latency_alert_warning(self):
        """Test warning latency alert."""
        monitor = PerformanceMonitor()
        
        with patch.object(monitor.logger, 'warning') as mock_warning:
            monitor.record_latency("test_stage", 1500.0)  # Above warning threshold
            
            mock_warning.assert_called_once()
            assert "WARNING" in mock_warning.call_args[0][0]
            assert "Elevated latency" in mock_warning.call_args[0][0]
    
    def test_queue_alert_critical(self):
        """Test critical queue size alert."""
        monitor = PerformanceMonitor()
        
        with patch.object(monitor.logger, 'warning') as mock_warning:
            monitor.record_queue_size("test_queue", 10)  # At critical threshold
            
            mock_warning.assert_called_once()
            assert "CRITICAL" in mock_warning.call_args[0][0]
            assert "Queue test_queue is full" in mock_warning.call_args[0][0]
    
    def test_alert_cooldown(self):
        """Test alert cooldown mechanism."""
        monitor = PerformanceMonitor()
        monitor._alert_cooldown = timedelta(seconds=1)  # Short cooldown for testing
        
        with patch.object(monitor.logger, 'warning') as mock_warning:
            # First alert should trigger
            monitor.record_latency("test_stage", 2500.0)
            assert mock_warning.call_count == 1
            
            # Second alert immediately should not trigger (cooldown)
            monitor.record_latency("test_stage", 2500.0)
            assert mock_warning.call_count == 1
            
            # Wait for cooldown and try again
            time.sleep(1.1)
            monitor.record_latency("test_stage", 2500.0)
            assert mock_warning.call_count == 2
    
    def test_reset_metrics(self):
        """Test resetting all metrics."""
        monitor = PerformanceMonitor()
        
        # Add some data
        monitor.record_latency("test_stage", 100.0)
        monitor.record_throughput("test_stage", 50.0)
        
        # Verify data exists
        assert len(monitor.collectors[MetricType.LATENCY].samples) == 1
        assert len(monitor.collectors[MetricType.THROUGHPUT].samples) == 1
        
        # Reset and verify data is cleared
        monitor.reset_metrics()
        
        for collector in monitor.collectors.values():
            assert len(collector.samples) == 0
        assert len(monitor._last_alerts) == 0
    
    def test_export_metrics_json(self):
        """Test exporting metrics in JSON format."""
        monitor = PerformanceMonitor()
        
        monitor.record_latency("test_stage", 100.0)
        
        json_export = monitor.export_metrics("json")
        
        # Should be valid JSON
        data = json.loads(json_export)
        assert "timestamp" in data
        assert "metrics" in data
        assert "latency" in data["metrics"]
    
    def test_export_metrics_csv(self):
        """Test exporting metrics in CSV format."""
        monitor = PerformanceMonitor()
        
        monitor.record_latency("test_stage", 100.0)
        
        csv_export = monitor.export_metrics("csv")
        
        lines = csv_export.split('\n')
        assert len(lines) >= 2  # Header + at least one data line
        assert "metric_type,stage,count,mean,median,min,max,std_dev,p95,p99" in lines[0]
    
    def test_export_metrics_invalid_format(self):
        """Test exporting metrics with invalid format."""
        monitor = PerformanceMonitor()
        
        with pytest.raises(ValueError) as exc_info:
            monitor.export_metrics("invalid_format")
        
        assert "Unsupported export format" in str(exc_info.value)


class TestLatencyTracker:
    """Test latency tracker context manager."""
    
    def test_latency_tracker_success(self):
        """Test latency tracker with successful operation."""
        monitor = PerformanceMonitor()
        
        with LatencyTracker(monitor, "test_stage", {"key": "value"}):
            time.sleep(0.01)  # Small delay
        
        collector = monitor.collectors[MetricType.LATENCY]
        assert len(collector.samples) == 1
        
        sample = collector.samples[0]
        assert sample.stage == "test_stage"
        assert sample.value > 0  # Should have some latency
        assert sample.metadata == {"key": "value"}
    
    def test_latency_tracker_with_exception(self):
        """Test latency tracker when exception occurs."""
        monitor = PerformanceMonitor()
        
        try:
            with LatencyTracker(monitor, "test_stage"):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Should still record latency
        latency_collector = monitor.collectors[MetricType.LATENCY]
        assert len(latency_collector.samples) == 1
        
        # Should also record error
        error_collector = monitor.collectors[MetricType.ERROR_RATE]
        assert len(error_collector.samples) == 1
        
        error_sample = error_collector.samples[0]
        assert error_sample.stage == "test_stage"
        assert error_sample.metadata["error_type"] == "ValueError"
        assert error_sample.metadata["error_message"] == "Test error"


if __name__ == "__main__":
    pytest.main([__file__])