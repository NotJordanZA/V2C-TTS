"""
System profiler for performance analysis and optimization.

This module provides comprehensive profiling capabilities to identify
bottlenecks and optimize system performance.
"""

import time
import psutil
import threading
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProfilerMetrics:
    """Metrics collected by the profiler."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: float = 0.0
    gpu_utilization: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0


@dataclass
class ComponentProfile:
    """Performance profile for a pipeline component."""
    component_name: str
    total_calls: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    p95_time_ms: float = 0.0
    p99_time_ms: float = 0.0
    error_count: int = 0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_timing(self, duration_ms: float, success: bool = True):
        """Add a timing measurement."""
        self.total_calls += 1
        if success:
            self.total_time_ms += duration_ms
            self.min_time_ms = min(self.min_time_ms, duration_ms)
            self.max_time_ms = max(self.max_time_ms, duration_ms)
            self.recent_times.append(duration_ms)
            
            # Update averages
            self.avg_time_ms = self.total_time_ms / max(1, self.total_calls - self.error_count)
            
            # Update percentiles if we have enough data
            if len(self.recent_times) >= 10:
                sorted_times = sorted(self.recent_times)
                self.p95_time_ms = sorted_times[int(len(sorted_times) * 0.95)]
                self.p99_time_ms = sorted_times[int(len(sorted_times) * 0.99)]
        else:
            self.error_count += 1


class SystemProfiler:
    """System-wide performance profiler."""
    
    def __init__(self, sampling_interval: float = 1.0):
        self.sampling_interval = sampling_interval
        self.is_profiling = False
        self.metrics_history: List[ProfilerMetrics] = []
        self.component_profiles: Dict[str, ComponentProfile] = {}
        self.profiling_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # GPU monitoring setup
        self.gpu_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_available = True
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except:
            logger.warning("GPU monitoring not available")
    
    def start_profiling(self):
        """Start system profiling."""
        if self.is_profiling:
            return
        
        self.is_profiling = True
        self.profiling_thread = threading.Thread(target=self._profiling_loop, daemon=True)
        self.profiling_thread.start()
        logger.info("System profiling started")
    
    def stop_profiling(self):
        """Stop system profiling."""
        self.is_profiling = False
        if self.profiling_thread:
            self.profiling_thread.join(timeout=2.0)
        logger.info("System profiling stopped")
    
    def _profiling_loop(self):
        """Main profiling loop."""
        process = psutil.Process()
        last_disk_io = process.io_counters()
        last_net_io = psutil.net_io_counters()
        
        while self.is_profiling:
            try:
                # Collect system metrics
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # Disk I/O
                current_disk_io = process.io_counters()
                disk_read_mb = (current_disk_io.read_bytes - last_disk_io.read_bytes) / 1024 / 1024
                disk_write_mb = (current_disk_io.write_bytes - last_disk_io.write_bytes) / 1024 / 1024
                last_disk_io = current_disk_io
                
                # Network I/O
                current_net_io = psutil.net_io_counters()
                if current_net_io and last_net_io:
                    net_sent_mb = (current_net_io.bytes_sent - last_net_io.bytes_sent) / 1024 / 1024
                    net_recv_mb = (current_net_io.bytes_recv - last_net_io.bytes_recv) / 1024 / 1024
                else:
                    net_sent_mb = net_recv_mb = 0.0
                last_net_io = current_net_io
                
                # GPU metrics
                gpu_memory_mb = 0.0
                gpu_utilization = 0.0
                if self.gpu_available:
                    try:
                        import pynvml
                        gpu_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                        gpu_memory_mb = gpu_info.used / 1024 / 1024
                        gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                        gpu_utilization = gpu_util.gpu
                    except:
                        pass
                
                # Store metrics
                metrics = ProfilerMetrics(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_mb=memory_mb,
                    gpu_memory_mb=gpu_memory_mb,
                    gpu_utilization=gpu_utilization,
                    disk_io_read_mb=disk_read_mb,
                    disk_io_write_mb=disk_write_mb,
                    network_sent_mb=net_sent_mb,
                    network_recv_mb=net_recv_mb
                )
                
                with self.lock:
                    self.metrics_history.append(metrics)
                    # Keep only last 1000 samples
                    if len(self.metrics_history) > 1000:
                        self.metrics_history.pop(0)
                
                time.sleep(self.sampling_interval)
                
            except Exception as e:
                logger.error(f"Error in profiling loop: {e}")
                time.sleep(self.sampling_interval)
    
    def profile_component(self, component_name: str):
        """Decorator to profile a component method."""
        def decorator(func):
            if asyncio.iscoroutinefunction(func):
                async def async_wrapper(*args, **kwargs):
                    start_time = time.time()
                    success = True
                    try:
                        result = await func(*args, **kwargs)
                        return result
                    except Exception as e:
                        success = False
                        raise
                    finally:
                        end_time = time.time()
                        duration_ms = (end_time - start_time) * 1000
                        self.add_component_timing(component_name, duration_ms, success)
                return async_wrapper
            else:
                def sync_wrapper(*args, **kwargs):
                    start_time = time.time()
                    success = True
                    try:
                        result = func(*args, **kwargs)
                        return result
                    except Exception as e:
                        success = False
                        raise
                    finally:
                        end_time = time.time()
                        duration_ms = (end_time - start_time) * 1000
                        self.add_component_timing(component_name, duration_ms, success)
                return sync_wrapper
        return decorator
    
    def add_component_timing(self, component_name: str, duration_ms: float, success: bool = True):
        """Add timing data for a component."""
        with self.lock:
            if component_name not in self.component_profiles:
                self.component_profiles[component_name] = ComponentProfile(component_name)
            
            self.component_profiles[component_name].add_timing(duration_ms, success)
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get system performance summary."""
        with self.lock:
            if not self.metrics_history:
                return {}
            
            recent_metrics = self.metrics_history[-60:]  # Last 60 samples
            
            return {
                'cpu': {
                    'avg_percent': statistics.mean(m.cpu_percent for m in recent_metrics),
                    'max_percent': max(m.cpu_percent for m in recent_metrics),
                    'current_percent': recent_metrics[-1].cpu_percent
                },
                'memory': {
                    'avg_mb': statistics.mean(m.memory_mb for m in recent_metrics),
                    'max_mb': max(m.memory_mb for m in recent_metrics),
                    'current_mb': recent_metrics[-1].memory_mb
                },
                'gpu': {
                    'avg_memory_mb': statistics.mean(m.gpu_memory_mb for m in recent_metrics),
                    'max_memory_mb': max(m.gpu_memory_mb for m in recent_metrics),
                    'avg_utilization': statistics.mean(m.gpu_utilization for m in recent_metrics),
                    'current_utilization': recent_metrics[-1].gpu_utilization
                },
                'disk_io': {
                    'avg_read_mb_per_sec': statistics.mean(m.disk_io_read_mb for m in recent_metrics),
                    'avg_write_mb_per_sec': statistics.mean(m.disk_io_write_mb for m in recent_metrics)
                }
            }
    
    def get_component_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get component performance summary."""
        with self.lock:
            summary = {}
            for name, profile in self.component_profiles.items():
                summary[name] = {
                    'total_calls': profile.total_calls,
                    'avg_time_ms': profile.avg_time_ms,
                    'min_time_ms': profile.min_time_ms if profile.min_time_ms != float('inf') else 0,
                    'max_time_ms': profile.max_time_ms,
                    'p95_time_ms': profile.p95_time_ms,
                    'p99_time_ms': profile.p99_time_ms,
                    'error_count': profile.error_count,
                    'success_rate': (profile.total_calls - profile.error_count) / max(1, profile.total_calls)
                }
            return summary
    
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # System bottlenecks
        system_summary = self.get_system_summary()
        if system_summary:
            if system_summary['cpu']['avg_percent'] > 80:
                bottlenecks.append({
                    'type': 'system',
                    'component': 'cpu',
                    'severity': 'high',
                    'description': f"High CPU usage: {system_summary['cpu']['avg_percent']:.1f}%",
                    'recommendation': 'Consider CPU optimization or load balancing'
                })
            
            if system_summary['memory']['current_mb'] > 4000:  # 4GB threshold
                bottlenecks.append({
                    'type': 'system',
                    'component': 'memory',
                    'severity': 'medium',
                    'description': f"High memory usage: {system_summary['memory']['current_mb']:.1f}MB",
                    'recommendation': 'Consider memory optimization or garbage collection'
                })
            
            if system_summary['gpu']['avg_utilization'] > 90:
                bottlenecks.append({
                    'type': 'system',
                    'component': 'gpu',
                    'severity': 'high',
                    'description': f"High GPU utilization: {system_summary['gpu']['avg_utilization']:.1f}%",
                    'recommendation': 'Consider GPU optimization or model quantization'
                })
        
        # Component bottlenecks
        component_summary = self.get_component_summary()
        for name, stats in component_summary.items():
            if stats['avg_time_ms'] > 500:  # 500ms threshold
                severity = 'high' if stats['avg_time_ms'] > 1000 else 'medium'
                bottlenecks.append({
                    'type': 'component',
                    'component': name,
                    'severity': severity,
                    'description': f"Slow component: {stats['avg_time_ms']:.1f}ms average",
                    'recommendation': f'Optimize {name} processing or consider caching'
                })
            
            if stats['success_rate'] < 0.95:
                bottlenecks.append({
                    'type': 'component',
                    'component': name,
                    'severity': 'high',
                    'description': f"High error rate: {(1-stats['success_rate'])*100:.1f}%",
                    'recommendation': f'Investigate and fix errors in {name}'
                })
        
        return bottlenecks
    
    def save_profile_report(self, output_file: str = "profile_report.json"):
        """Save profiling report to file."""
        report = {
            'timestamp': time.time(),
            'profiling_duration_seconds': len(self.metrics_history) * self.sampling_interval,
            'system_summary': self.get_system_summary(),
            'component_summary': self.get_component_summary(),
            'bottlenecks': self.identify_bottlenecks(),
            'raw_metrics': [
                {
                    'timestamp': m.timestamp,
                    'cpu_percent': m.cpu_percent,
                    'memory_mb': m.memory_mb,
                    'gpu_memory_mb': m.gpu_memory_mb,
                    'gpu_utilization': m.gpu_utilization
                }
                for m in self.metrics_history[-100:]  # Last 100 samples
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Profile report saved to {output_file}")
        return report


class OptimizationManager:
    """Manages system optimizations based on profiling data."""
    
    def __init__(self, profiler: SystemProfiler):
        self.profiler = profiler
        self.optimizations_applied = []
        self.quality_settings = {
            'stt_model_size': 'base',
            'llm_context_length': 2048,
            'tts_quality': 'medium',
            'audio_sample_rate': 16000,
            'chunk_size': 1024
        }
    
    def analyze_and_optimize(self) -> List[Dict[str, Any]]:
        """Analyze system performance and apply optimizations."""
        bottlenecks = self.profiler.identify_bottlenecks()
        optimizations = []
        
        for bottleneck in bottlenecks:
            optimization = self._get_optimization_for_bottleneck(bottleneck)
            if optimization:
                optimizations.append(optimization)
        
        return optimizations
    
    def _get_optimization_for_bottleneck(self, bottleneck: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get optimization recommendation for a specific bottleneck."""
        component = bottleneck['component']
        severity = bottleneck['severity']
        
        if component == 'cpu' and severity == 'high':
            return {
                'type': 'quality_reduction',
                'target': 'stt_model_size',
                'current_value': self.quality_settings['stt_model_size'],
                'recommended_value': 'tiny',
                'description': 'Reduce STT model size to decrease CPU usage',
                'impact': 'Lower transcription accuracy but faster processing'
            }
        
        elif component == 'memory' and severity in ['high', 'medium']:
            return {
                'type': 'memory_optimization',
                'target': 'llm_context_length',
                'current_value': self.quality_settings['llm_context_length'],
                'recommended_value': 1024,
                'description': 'Reduce LLM context length to save memory',
                'impact': 'Less context awareness but lower memory usage'
            }
        
        elif component == 'gpu' and severity == 'high':
            return {
                'type': 'gpu_optimization',
                'target': 'model_quantization',
                'description': 'Apply model quantization to reduce GPU memory usage',
                'impact': 'Slightly lower quality but significantly less GPU memory'
            }
        
        elif 'stt' in component.lower():
            return {
                'type': 'component_optimization',
                'target': 'stt_batching',
                'description': 'Enable audio batching for STT processing',
                'impact': 'Better throughput but slightly higher latency'
            }
        
        elif 'tts' in component.lower():
            return {
                'type': 'quality_reduction',
                'target': 'tts_quality',
                'current_value': self.quality_settings['tts_quality'],
                'recommended_value': 'low',
                'description': 'Reduce TTS quality to improve speed',
                'impact': 'Lower audio quality but faster generation'
            }
        
        return None
    
    def apply_optimization(self, optimization: Dict[str, Any]) -> bool:
        """Apply a specific optimization."""
        try:
            opt_type = optimization['type']
            
            if opt_type == 'quality_reduction':
                target = optimization['target']
                new_value = optimization['recommended_value']
                self.quality_settings[target] = new_value
                
            elif opt_type == 'memory_optimization':
                target = optimization['target']
                new_value = optimization['recommended_value']
                self.quality_settings[target] = new_value
            
            self.optimizations_applied.append({
                'timestamp': time.time(),
                'optimization': optimization,
                'status': 'applied'
            })
            
            logger.info(f"Applied optimization: {optimization['description']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply optimization: {e}")
            self.optimizations_applied.append({
                'timestamp': time.time(),
                'optimization': optimization,
                'status': 'failed',
                'error': str(e)
            })
            return False
    
    def get_current_settings(self) -> Dict[str, Any]:
        """Get current quality settings."""
        return self.quality_settings.copy()
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of applied optimizations."""
        return self.optimizations_applied.copy()


# Global profiler instance
_global_profiler: Optional[SystemProfiler] = None


def get_profiler() -> SystemProfiler:
    """Get the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = SystemProfiler()
    return _global_profiler


def profile_component(component_name: str):
    """Decorator to profile a component."""
    return get_profiler().profile_component(component_name)