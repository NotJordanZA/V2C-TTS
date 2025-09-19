"""
Dynamic quality adjustment system for performance optimization.

This module automatically adjusts quality settings based on system performance
to maintain optimal user experience while meeting latency requirements.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
import statistics

from .profiler import SystemProfiler, OptimizationManager
from .interfaces import PipelineConfig

logger = logging.getLogger(__name__)


class QualityLevel(str, Enum):
    """Quality levels for different components."""
    ULTRA = "ultra"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


@dataclass
class QualityProfile:
    """Quality profile configuration."""
    name: str
    stt_model_size: str
    llm_max_tokens: int
    tts_quality: str
    audio_sample_rate: int
    chunk_size: int
    max_latency_ms: int
    description: str


class QualityManager:
    """Manages dynamic quality adjustments based on system performance."""
    
    def __init__(self, profiler: SystemProfiler, initial_config: PipelineConfig):
        self.profiler = profiler
        self.config = initial_config
        self.current_quality_level = QualityLevel.HIGH
        self.adjustment_callbacks: List[Callable] = []
        self.performance_history = []
        self.adjustment_cooldown = 10.0  # seconds
        self.last_adjustment_time = 0.0
        
        # Define quality profiles
        self.quality_profiles = {
            QualityLevel.ULTRA: QualityProfile(
                name="Ultra Quality",
                stt_model_size="large",
                llm_max_tokens=4096,
                tts_quality="high",
                audio_sample_rate=22050,
                chunk_size=2048,
                max_latency_ms=3000,
                description="Maximum quality, highest resource usage"
            ),
            QualityLevel.HIGH: QualityProfile(
                name="High Quality",
                stt_model_size="base",
                llm_max_tokens=2048,
                tts_quality="medium",
                audio_sample_rate=16000,
                chunk_size=1024,
                max_latency_ms=2000,
                description="High quality with good performance"
            ),
            QualityLevel.MEDIUM: QualityProfile(
                name="Medium Quality",
                stt_model_size="small",
                llm_max_tokens=1024,
                tts_quality="medium",
                audio_sample_rate=16000,
                chunk_size=512,
                max_latency_ms=1500,
                description="Balanced quality and performance"
            ),
            QualityLevel.LOW: QualityProfile(
                name="Low Quality",
                stt_model_size="tiny",
                llm_max_tokens=512,
                tts_quality="low",
                audio_sample_rate=8000,
                chunk_size=256,
                max_latency_ms=1000,
                description="Lower quality for better performance"
            ),
            QualityLevel.MINIMAL: QualityProfile(
                name="Minimal Quality",
                stt_model_size="tiny",
                llm_max_tokens=256,
                tts_quality="low",
                audio_sample_rate=8000,
                chunk_size=128,
                max_latency_ms=500,
                description="Minimal quality for maximum performance"
            )
        }
        
        # Performance thresholds for quality adjustment
        self.performance_thresholds = {
            'cpu_high': 85.0,
            'cpu_critical': 95.0,
            'memory_high_mb': 4000,
            'memory_critical_mb': 6000,
            'gpu_high': 90.0,
            'gpu_critical': 98.0,
            'latency_high_ms': 2500,
            'latency_critical_ms': 4000
        }
    
    def register_adjustment_callback(self, callback: Callable[[QualityProfile], None]):
        """Register a callback to be called when quality is adjusted."""
        self.adjustment_callbacks.append(callback)
    
    async def start_monitoring(self):
        """Start continuous quality monitoring and adjustment."""
        logger.info("Starting quality monitoring")
        
        while True:
            try:
                await self._check_and_adjust_quality()
                await asyncio.sleep(5.0)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Error in quality monitoring: {e}")
                await asyncio.sleep(5.0)
    
    async def _check_and_adjust_quality(self):
        """Check system performance and adjust quality if needed."""
        # Get current system performance
        system_summary = self.profiler.get_system_summary()
        component_summary = self.profiler.get_component_summary()
        
        if not system_summary:
            return
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(system_summary, component_summary)
        
        # Store performance history
        self.performance_history.append({
            'timestamp': time.time(),
            'score': performance_score,
            'system': system_summary,
            'components': component_summary
        })
        
        # Keep only recent history
        if len(self.performance_history) > 60:  # 5 minutes at 5-second intervals
            self.performance_history.pop(0)
        
        # Check if adjustment is needed
        current_time = time.time()
        if current_time - self.last_adjustment_time < self.adjustment_cooldown:
            return  # Still in cooldown period
        
        new_quality_level = self._determine_optimal_quality_level(performance_score, system_summary)
        
        if new_quality_level != self.current_quality_level:
            await self._adjust_quality(new_quality_level)
            self.last_adjustment_time = current_time
    
    def _calculate_performance_score(self, system_summary: Dict[str, Any], component_summary: Dict[str, Any]) -> float:
        """Calculate overall performance score (0-100, higher is better)."""
        score = 100.0
        
        # CPU performance impact
        cpu_usage = system_summary.get('cpu', {}).get('avg_percent', 0)
        if cpu_usage > self.performance_thresholds['cpu_critical']:
            score -= 40
        elif cpu_usage > self.performance_thresholds['cpu_high']:
            score -= 20
        
        # Memory performance impact
        memory_usage = system_summary.get('memory', {}).get('current_mb', 0)
        if memory_usage > self.performance_thresholds['memory_critical_mb']:
            score -= 30
        elif memory_usage > self.performance_thresholds['memory_high_mb']:
            score -= 15
        
        # GPU performance impact
        gpu_usage = system_summary.get('gpu', {}).get('avg_utilization', 0)
        if gpu_usage > self.performance_thresholds['gpu_critical']:
            score -= 35
        elif gpu_usage > self.performance_thresholds['gpu_high']:
            score -= 15
        
        # Component latency impact
        for component_name, stats in component_summary.items():
            avg_latency = stats.get('avg_time_ms', 0)
            if avg_latency > 1000:  # 1 second
                score -= 20
            elif avg_latency > 500:  # 500ms
                score -= 10
            
            # Error rate impact
            success_rate = stats.get('success_rate', 1.0)
            if success_rate < 0.9:
                score -= 25
            elif success_rate < 0.95:
                score -= 10
        
        return max(0, score)
    
    def _determine_optimal_quality_level(self, performance_score: float, system_summary: Dict[str, Any]) -> QualityLevel:
        """Determine optimal quality level based on performance."""
        # Get recent performance trend
        if len(self.performance_history) >= 3:
            recent_scores = [h['score'] for h in self.performance_history[-3:]]
            trend = statistics.mean(recent_scores)
        else:
            trend = performance_score
        
        # Determine quality level based on performance score and trend
        if trend >= 80:
            # System performing well, can use high quality
            if self.current_quality_level in [QualityLevel.LOW, QualityLevel.MINIMAL]:
                return QualityLevel.MEDIUM  # Gradual increase
            elif self.current_quality_level == QualityLevel.MEDIUM:
                return QualityLevel.HIGH
            else:
                return QualityLevel.HIGH
        
        elif trend >= 60:
            # Moderate performance, use medium quality
            return QualityLevel.MEDIUM
        
        elif trend >= 40:
            # Poor performance, reduce to low quality
            return QualityLevel.LOW
        
        else:
            # Critical performance issues, use minimal quality
            return QualityLevel.MINIMAL
    
    async def _adjust_quality(self, new_quality_level: QualityLevel):
        """Adjust system quality to the specified level."""
        old_level = self.current_quality_level
        self.current_quality_level = new_quality_level
        
        profile = self.quality_profiles[new_quality_level]
        
        logger.info(f"Adjusting quality from {old_level.value} to {new_quality_level.value}")
        logger.info(f"New profile: {profile.description}")
        
        # Update configuration
        self.config.stt_model_size = profile.stt_model_size
        self.config.sample_rate = profile.audio_sample_rate
        self.config.chunk_size = profile.chunk_size
        self.config.max_latency_ms = profile.max_latency_ms
        
        # Notify callbacks
        for callback in self.adjustment_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(profile)
                else:
                    callback(profile)
            except Exception as e:
                logger.error(f"Error in quality adjustment callback: {e}")
    
    def force_quality_level(self, quality_level: QualityLevel):
        """Force a specific quality level (bypasses automatic adjustment)."""
        self.current_quality_level = quality_level
        profile = self.quality_profiles[quality_level]
        
        # Update configuration
        self.config.stt_model_size = profile.stt_model_size
        self.config.sample_rate = profile.audio_sample_rate
        self.config.chunk_size = profile.chunk_size
        self.config.max_latency_ms = profile.max_latency_ms
        
        logger.info(f"Forced quality level to {quality_level.value}")
        
        # Reset adjustment cooldown to prevent immediate changes
        self.last_adjustment_time = time.time()
    
    def get_current_quality_info(self) -> Dict[str, Any]:
        """Get information about current quality settings."""
        profile = self.quality_profiles[self.current_quality_level]
        
        # Get recent performance data
        recent_performance = None
        if self.performance_history:
            recent_performance = self.performance_history[-1]
        
        return {
            'current_level': self.current_quality_level.value,
            'profile': {
                'name': profile.name,
                'description': profile.description,
                'stt_model_size': profile.stt_model_size,
                'llm_max_tokens': profile.llm_max_tokens,
                'tts_quality': profile.tts_quality,
                'audio_sample_rate': profile.audio_sample_rate,
                'chunk_size': profile.chunk_size,
                'max_latency_ms': profile.max_latency_ms
            },
            'recent_performance_score': recent_performance['score'] if recent_performance else None,
            'adjustment_history': [
                {
                    'timestamp': h['timestamp'],
                    'score': h['score']
                }
                for h in self.performance_history[-10:]  # Last 10 measurements
            ]
        }
    
    def get_available_quality_levels(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all available quality levels."""
        return {
            level.value: {
                'name': profile.name,
                'description': profile.description,
                'stt_model_size': profile.stt_model_size,
                'max_latency_ms': profile.max_latency_ms
            }
            for level, profile in self.quality_profiles.items()
        }
    
    def get_performance_recommendations(self) -> List[Dict[str, Any]]:
        """Get performance improvement recommendations."""
        recommendations = []
        
        if not self.performance_history:
            return recommendations
        
        recent_performance = self.performance_history[-1]
        system_summary = recent_performance['system']
        
        # CPU recommendations
        cpu_usage = system_summary.get('cpu', {}).get('avg_percent', 0)
        if cpu_usage > 80:
            recommendations.append({
                'type': 'cpu_optimization',
                'severity': 'high' if cpu_usage > 90 else 'medium',
                'description': f'High CPU usage ({cpu_usage:.1f}%)',
                'suggestions': [
                    'Reduce STT model size',
                    'Enable CPU-specific optimizations',
                    'Close other applications'
                ]
            })
        
        # Memory recommendations
        memory_usage = system_summary.get('memory', {}).get('current_mb', 0)
        if memory_usage > 3000:
            recommendations.append({
                'type': 'memory_optimization',
                'severity': 'high' if memory_usage > 5000 else 'medium',
                'description': f'High memory usage ({memory_usage:.0f}MB)',
                'suggestions': [
                    'Reduce LLM context length',
                    'Enable model quantization',
                    'Clear audio buffers more frequently'
                ]
            })
        
        # GPU recommendations
        gpu_usage = system_summary.get('gpu', {}).get('avg_utilization', 0)
        if gpu_usage > 85:
            recommendations.append({
                'type': 'gpu_optimization',
                'severity': 'high' if gpu_usage > 95 else 'medium',
                'description': f'High GPU usage ({gpu_usage:.1f}%)',
                'suggestions': [
                    'Enable model quantization',
                    'Reduce batch sizes',
                    'Use mixed precision training'
                ]
            })
        
        return recommendations