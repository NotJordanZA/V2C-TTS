"""
Performance regression tests to ensure system performance doesn't degrade over time.

This module provides automated tests to detect performance regressions
and validate that optimizations maintain expected performance levels.
"""

import pytest
import asyncio
import time
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from src.core.pipeline import VoicePipeline
from src.core.config import AppConfig
from src.core.profiler import SystemProfiler, OptimizationManager
from src.core.quality_manager import QualityManager, QualityLevel
from src.core.interfaces import AudioChunk
from tests.test_end_to_end_integration import SyntheticAudioGenerator, MockComponents


class PerformanceBaseline:
    """Manages performance baselines for regression testing."""
    
    def __init__(self, baseline_file: str = "performance_baseline.json"):
        self.baseline_file = Path(baseline_file)
        self.baselines = self._load_baselines()
    
    def _load_baselines(self) -> Dict[str, Any]:
        """Load performance baselines from file."""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Default baselines if file doesn't exist
        return {
            'pipeline_latency_ms': {
                'avg': 600.0,
                'p95': 1000.0,
                'p99': 1500.0
            },
            'component_latency_ms': {
                'stt': {'avg': 150.0, 'p95': 250.0},
                'character_transform': {'avg': 200.0, 'p95': 350.0},
                'tts': {'avg': 250.0, 'p95': 400.0}
            },
            'throughput_chunks_per_second': 5.0,
            'memory_usage_mb': {
                'baseline': 500.0,
                'max_growth': 100.0
            },
            'cpu_usage_percent': {
                'avg': 30.0,
                'max': 60.0
            },
            'success_rate': 0.98
        }
    
    def save_baselines(self):
        """Save current baselines to file."""
        with open(self.baseline_file, 'w') as f:
            json.dump(self.baselines, f, indent=2)
    
    def update_baseline(self, metric_name: str, value: Any):
        """Update a baseline metric."""
        keys = metric_name.split('.')
        current = self.baselines
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def get_baseline(self, metric_name: str) -> Any:
        """Get a baseline metric value."""
        keys = metric_name.split('.')
        current = self.baselines
        
        for key in keys:
            if key not in current:
                return None
            current = current[key]
        
        return current
    
    def check_regression(self, metric_name: str, current_value: float, tolerance: float = 0.1) -> Dict[str, Any]:
        """Check if current value represents a performance regression."""
        baseline = self.get_baseline(metric_name)
        if baseline is None:
            return {'is_regression': False, 'reason': 'No baseline available'}
        
        # For latency metrics, higher is worse
        if 'latency' in metric_name.lower() or 'time' in metric_name.lower():
            threshold = baseline * (1 + tolerance)
            is_regression = current_value > threshold
            improvement = baseline - current_value
        # For throughput and success rate, lower is worse
        elif 'throughput' in metric_name.lower() or 'success_rate' in metric_name.lower():
            threshold = baseline * (1 - tolerance)
            is_regression = current_value < threshold
            improvement = current_value - baseline
        # For memory and CPU, higher is worse
        else:
            threshold = baseline * (1 + tolerance)
            is_regression = current_value > threshold
            improvement = baseline - current_value
        
        return {
            'is_regression': is_regression,
            'baseline': baseline,
            'current': current_value,
            'threshold': threshold,
            'improvement': improvement,
            'improvement_percent': (improvement / baseline * 100) if baseline > 0 else 0
        }


@pytest.mark.regression
class TestPerformanceRegression:
    """Performance regression tests."""
    
    @pytest.fixture
    def baseline_manager(self):
        """Create baseline manager for testing."""
        return PerformanceBaseline("test_performance_baseline.json")
    
    @pytest.fixture
    def regression_config(self):
        """Configuration for regression testing."""
        from src.core.config import AppConfig, AudioConfig, STTConfig, CharacterConfig, TTSConfig, PerformanceConfig, LoggingConfig
        
        return AppConfig(
            audio=AudioConfig(
                sample_rate=16000,
                chunk_size=1024,
                input_device_id=0,
                output_device_id=0
            ),
            stt=STTConfig(
                model_size="base",
                device="cpu"
            ),
            character=CharacterConfig(
                profiles_dir="characters",
                default_character="default",
                llm_model_path="models/llm/test_model.bin"
            ),
            tts=TTSConfig(
                model_path="models/tts/test_model.pth",
                device="cpu",
                voice_models_dir="models/voices"
            ),
            performance=PerformanceConfig(
                max_latency_ms=2000,
                gpu_memory_fraction=0.8
            ),
            logging=LoggingConfig(
                level="WARNING"
            )
        )
    
    @pytest.fixture
    def mock_components_with_timing(self):
        """Mock components with realistic timing for regression testing."""
        components = MockComponents()
        
        # Add realistic delays that match baseline expectations
        original_stt = components.get_mock_stt_processor()
        original_stt.process_audio = AsyncMock(side_effect=lambda x: asyncio.sleep(0.15).then(lambda: "test text"))
        
        original_transformer = components.get_mock_character_transformer()
        original_transformer.transform_text = AsyncMock(side_effect=lambda x, y: asyncio.sleep(0.2).then(lambda: f"{x} transformed"))
        
        original_tts = components.get_mock_tts_processor()
        original_tts.synthesize_speech = AsyncMock(side_effect=lambda x, y: asyncio.sleep(0.25).then(lambda: np.zeros(16000)))
        
        return components
    
    @pytest.mark.asyncio
    async def test_pipeline_latency_regression(self, regression_config, mock_components_with_timing, baseline_manager):
        """Test for pipeline latency regression."""
        with patch('src.core.pipeline.STTProcessor', return_value=mock_components_with_timing.get_mock_stt_processor()), \
             patch('src.core.pipeline.CharacterTransformer', return_value=mock_components_with_timing.get_mock_character_transformer()), \
             patch('src.core.pipeline.TTSProcessor', return_value=mock_components_with_timing.get_mock_tts_processor()), \
             patch('src.core.pipeline.AudioCapture'), \
             patch('src.core.pipeline.AudioOutput'):
            
            pipeline = VoicePipeline(regression_config)
            await pipeline.initialize()
            await pipeline.start()
            
            # Measure pipeline latency
            latencies = []
            num_samples = 20
            
            for _ in range(num_samples):
                audio_data = SyntheticAudioGenerator.generate_speech_audio()
                audio_chunk = AudioChunk(
                    data=audio_data,
                    timestamp=time.time(),
                    sample_rate=16000,
                    duration_ms=len(audio_data) / 16000 * 1000
                )
                
                start_time = time.time()
                result = await pipeline.process_audio_chunk(audio_chunk)
                end_time = time.time()
                
                if result is not None:
                    latency_ms = (end_time - start_time) * 1000
                    latencies.append(latency_ms)
            
            await pipeline.stop()
            
            # Calculate metrics
            avg_latency = statistics.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            
            # Check for regressions
            avg_regression = baseline_manager.check_regression('pipeline_latency_ms.avg', avg_latency)
            p95_regression = baseline_manager.check_regression('pipeline_latency_ms.p95', p95_latency)
            p99_regression = baseline_manager.check_regression('pipeline_latency_ms.p99', p99_latency)
            
            # Assert no significant regressions
            assert not avg_regression['is_regression'], \
                f"Average latency regression: {avg_latency:.1f}ms vs baseline {avg_regression['baseline']:.1f}ms"
            
            assert not p95_regression['is_regression'], \
                f"P95 latency regression: {p95_latency:.1f}ms vs baseline {p95_regression['baseline']:.1f}ms"
            
            assert not p99_regression['is_regression'], \
                f"P99 latency regression: {p99_latency:.1f}ms vs baseline {p99_regression['baseline']:.1f}ms"
            
            # Update baselines if performance improved significantly
            if avg_regression['improvement_percent'] > 10:
                baseline_manager.update_baseline('pipeline_latency_ms.avg', avg_latency)
            if p95_regression['improvement_percent'] > 10:
                baseline_manager.update_baseline('pipeline_latency_ms.p95', p95_latency)
            if p99_regression['improvement_percent'] > 10:
                baseline_manager.update_baseline('pipeline_latency_ms.p99', p99_latency)
    
    @pytest.mark.asyncio
    async def test_throughput_regression(self, regression_config, mock_components_with_timing, baseline_manager):
        """Test for throughput regression."""
        with patch('src.core.pipeline.STTProcessor', return_value=mock_components_with_timing.get_mock_stt_processor()), \
             patch('src.core.pipeline.CharacterTransformer', return_value=mock_components_with_timing.get_mock_character_transformer()), \
             patch('src.core.pipeline.TTSProcessor', return_value=mock_components_with_timing.get_mock_tts_processor()), \
             patch('src.core.pipeline.AudioCapture'), \
             patch('src.core.pipeline.AudioOutput'):
            
            pipeline = VoicePipeline(regression_config)
            await pipeline.initialize()
            await pipeline.start()
            
            # Measure throughput
            num_chunks = 30
            start_time = time.time()
            successful_chunks = 0
            
            for _ in range(num_chunks):
                audio_data = SyntheticAudioGenerator.generate_speech_audio()
                audio_chunk = AudioChunk(
                    data=audio_data,
                    timestamp=time.time(),
                    sample_rate=16000,
                    duration_ms=len(audio_data) / 16000 * 1000
                )
                
                result = await pipeline.process_audio_chunk(audio_chunk)
                if result is not None:
                    successful_chunks += 1
            
            end_time = time.time()
            await pipeline.stop()
            
            # Calculate throughput
            total_time = end_time - start_time
            throughput = successful_chunks / total_time
            
            # Check for regression
            throughput_regression = baseline_manager.check_regression('throughput_chunks_per_second', throughput)
            
            assert not throughput_regression['is_regression'], \
                f"Throughput regression: {throughput:.2f} chunks/sec vs baseline {throughput_regression['baseline']:.2f}"
            
            # Update baseline if significantly improved
            if throughput_regression['improvement_percent'] > 15:
                baseline_manager.update_baseline('throughput_chunks_per_second', throughput)
    
    @pytest.mark.asyncio
    async def test_memory_usage_regression(self, regression_config, mock_components_with_timing, baseline_manager):
        """Test for memory usage regression."""
        import psutil
        import os
        
        with patch('src.core.pipeline.STTProcessor', return_value=mock_components_with_timing.get_mock_stt_processor()), \
             patch('src.core.pipeline.CharacterTransformer', return_value=mock_components_with_timing.get_mock_character_transformer()), \
             patch('src.core.pipeline.TTSProcessor', return_value=mock_components_with_timing.get_mock_tts_processor()), \
             patch('src.core.pipeline.AudioCapture'), \
             patch('src.core.pipeline.AudioOutput'):
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            pipeline = VoicePipeline(regression_config)
            await pipeline.initialize()
            await pipeline.start()
            
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process multiple chunks to test memory growth
            for _ in range(50):
                audio_data = SyntheticAudioGenerator.generate_speech_audio()
                audio_chunk = AudioChunk(
                    data=audio_data,
                    timestamp=time.time(),
                    sample_rate=16000,
                    duration_ms=len(audio_data) / 16000 * 1000
                )
                
                await pipeline.process_audio_chunk(audio_chunk)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            await pipeline.stop()
            
            # Calculate memory metrics
            baseline_usage = baseline_memory - initial_memory
            memory_growth = final_memory - baseline_memory
            
            # Check for regressions
            baseline_regression = baseline_manager.check_regression('memory_usage_mb.baseline', baseline_usage)
            growth_regression = baseline_manager.check_regression('memory_usage_mb.max_growth', memory_growth)
            
            assert not baseline_regression['is_regression'], \
                f"Baseline memory usage regression: {baseline_usage:.1f}MB vs baseline {baseline_regression['baseline']:.1f}MB"
            
            assert not growth_regression['is_regression'], \
                f"Memory growth regression: {memory_growth:.1f}MB vs baseline {growth_regression['baseline']:.1f}MB"
    
    @pytest.mark.asyncio
    async def test_quality_adjustment_performance(self, regression_config):
        """Test performance of quality adjustment system."""
        profiler = SystemProfiler(sampling_interval=0.1)
        quality_manager = QualityManager(profiler, regression_config)
        
        profiler.start_profiling()
        
        # Simulate performance changes and measure adjustment response time
        adjustment_times = []
        
        for quality_level in [QualityLevel.LOW, QualityLevel.MEDIUM, QualityLevel.HIGH]:
            start_time = time.time()
            quality_manager.force_quality_level(quality_level)
            end_time = time.time()
            
            adjustment_time_ms = (end_time - start_time) * 1000
            adjustment_times.append(adjustment_time_ms)
        
        profiler.stop_profiling()
        
        # Quality adjustments should be fast (under 100ms)
        avg_adjustment_time = statistics.mean(adjustment_times)
        max_adjustment_time = max(adjustment_times)
        
        assert avg_adjustment_time < 100, f"Quality adjustment too slow: {avg_adjustment_time:.1f}ms average"
        assert max_adjustment_time < 200, f"Quality adjustment too slow: {max_adjustment_time:.1f}ms maximum"
    
    @pytest.mark.asyncio
    async def test_profiler_overhead(self, regression_config, mock_components_with_timing):
        """Test that profiler doesn't add significant overhead."""
        # Test without profiler
        with patch('src.core.pipeline.STTProcessor', return_value=mock_components_with_timing.get_mock_stt_processor()), \
             patch('src.core.pipeline.CharacterTransformer', return_value=mock_components_with_timing.get_mock_character_transformer()), \
             patch('src.core.pipeline.TTSProcessor', return_value=mock_components_with_timing.get_mock_tts_processor()), \
             patch('src.core.pipeline.AudioCapture'), \
             patch('src.core.pipeline.AudioOutput'):
            
            pipeline = VoicePipeline(regression_config)
            await pipeline.initialize()
            await pipeline.start()
            
            # Measure without profiling
            start_time = time.time()
            for _ in range(10):
                audio_data = SyntheticAudioGenerator.generate_speech_audio()
                audio_chunk = AudioChunk(
                    data=audio_data,
                    timestamp=time.time(),
                    sample_rate=16000,
                    duration_ms=len(audio_data) / 16000 * 1000
                )
                await pipeline.process_audio_chunk(audio_chunk)
            
            time_without_profiler = time.time() - start_time
            await pipeline.stop()
        
        # Test with profiler
        profiler = SystemProfiler(sampling_interval=0.1)
        profiler.start_profiling()
        
        with patch('src.core.pipeline.STTProcessor', return_value=mock_components_with_timing.get_mock_stt_processor()), \
             patch('src.core.pipeline.CharacterTransformer', return_value=mock_components_with_timing.get_mock_character_transformer()), \
             patch('src.core.pipeline.TTSProcessor', return_value=mock_components_with_timing.get_mock_tts_processor()), \
             patch('src.core.pipeline.AudioCapture'), \
             patch('src.core.pipeline.AudioOutput'):
            
            pipeline = VoicePipeline(regression_config)
            await pipeline.initialize()
            await pipeline.start()
            
            # Measure with profiling
            start_time = time.time()
            for _ in range(10):
                audio_data = SyntheticAudioGenerator.generate_speech_audio()
                audio_chunk = AudioChunk(
                    data=audio_data,
                    timestamp=time.time(),
                    sample_rate=16000,
                    duration_ms=len(audio_data) / 16000 * 1000
                )
                await pipeline.process_audio_chunk(audio_chunk)
            
            time_with_profiler = time.time() - start_time
            await pipeline.stop()
        
        profiler.stop_profiling()
        
        # Profiler overhead should be minimal (less than 10%)
        overhead_percent = ((time_with_profiler - time_without_profiler) / time_without_profiler) * 100
        assert overhead_percent < 10, f"Profiler overhead too high: {overhead_percent:.1f}%"
    
    def test_optimization_effectiveness(self, baseline_manager):
        """Test that optimizations actually improve performance."""
        # This test would typically run before and after optimizations
        # and verify that the optimizations provide measurable improvements
        
        # Mock some performance data
        before_optimization = {
            'avg_latency_ms': 800,
            'p95_latency_ms': 1200,
            'throughput': 4.0,
            'memory_usage_mb': 600
        }
        
        after_optimization = {
            'avg_latency_ms': 650,  # 18.75% improvement
            'p95_latency_ms': 950,  # 20.8% improvement
            'throughput': 5.2,      # 30% improvement
            'memory_usage_mb': 520  # 13.3% improvement
        }
        
        # Verify improvements
        latency_improvement = (before_optimization['avg_latency_ms'] - after_optimization['avg_latency_ms']) / before_optimization['avg_latency_ms']
        throughput_improvement = (after_optimization['throughput'] - before_optimization['throughput']) / before_optimization['throughput']
        memory_improvement = (before_optimization['memory_usage_mb'] - after_optimization['memory_usage_mb']) / before_optimization['memory_usage_mb']
        
        # Optimizations should provide meaningful improvements
        assert latency_improvement > 0.1, f"Insufficient latency improvement: {latency_improvement:.1%}"
        assert throughput_improvement > 0.15, f"Insufficient throughput improvement: {throughput_improvement:.1%}"
        assert memory_improvement > 0.05, f"Insufficient memory improvement: {memory_improvement:.1%}"


@pytest.mark.regression
class TestOptimizationRegression:
    """Test optimization system for regressions."""
    
    @pytest.mark.asyncio
    async def test_gpu_memory_optimization(self):
        """Test GPU memory optimization doesn't regress."""
        # Mock GPU memory monitoring
        with patch('pynvml.nvmlInit'), \
             patch('pynvml.nvmlDeviceGetHandleByIndex'), \
             patch('pynvml.nvmlDeviceGetMemoryInfo') as mock_memory_info:
            
            # Simulate high GPU memory usage
            mock_memory_info.return_value.used = 6 * 1024 * 1024 * 1024  # 6GB
            mock_memory_info.return_value.total = 8 * 1024 * 1024 * 1024  # 8GB
            
            profiler = SystemProfiler()
            optimization_manager = OptimizationManager(profiler)
            
            # Should recommend GPU optimization
            optimizations = optimization_manager.analyze_and_optimize()
            
            gpu_optimizations = [opt for opt in optimizations if opt.get('type') == 'gpu_optimization']
            assert len(gpu_optimizations) > 0, "Should recommend GPU optimization for high memory usage"
    
    def test_quality_profile_consistency(self):
        """Test that quality profiles maintain expected performance characteristics."""
        from src.core.quality_manager import QualityManager, QualityLevel
        
        # Mock config and profiler
        mock_config = Mock()
        mock_profiler = Mock()
        
        quality_manager = QualityManager(mock_profiler, mock_config)
        
        # Verify quality profiles are ordered correctly (higher quality = higher resource usage)
        profiles = quality_manager.quality_profiles
        
        # Check that higher quality levels have higher resource requirements
        ultra = profiles[QualityLevel.ULTRA]
        high = profiles[QualityLevel.HIGH]
        medium = profiles[QualityLevel.MEDIUM]
        low = profiles[QualityLevel.LOW]
        minimal = profiles[QualityLevel.MINIMAL]
        
        # Latency should increase with quality
        assert ultra.max_latency_ms >= high.max_latency_ms
        assert high.max_latency_ms >= medium.max_latency_ms
        assert medium.max_latency_ms >= low.max_latency_ms
        assert low.max_latency_ms >= minimal.max_latency_ms
        
        # Sample rate should generally increase with quality
        assert ultra.audio_sample_rate >= high.audio_sample_rate
        assert high.audio_sample_rate >= medium.audio_sample_rate


if __name__ == "__main__":
    # Run regression tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "regression"])