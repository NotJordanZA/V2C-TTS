"""
Performance benchmarking tests for the voice character transformation pipeline.

This module provides comprehensive performance testing and benchmarking
to ensure the system meets latency and throughput requirements.
"""

import pytest
import asyncio
import time
import statistics
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from unittest.mock import Mock, patch, AsyncMock
import json
from pathlib import Path

from src.core.pipeline import VoicePipeline, PipelineMetrics
from src.core.config import AppConfig
from src.core.interfaces import AudioChunk
from tests.test_end_to_end_integration import SyntheticAudioGenerator, MockComponents


@dataclass
class BenchmarkResult:
    """Results from a performance benchmark test."""
    test_name: str
    num_samples: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_chunks_per_second: float
    success_rate: float
    memory_usage_mb: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert benchmark result to dictionary."""
        return {
            "test_name": self.test_name,
            "num_samples": self.num_samples,
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "throughput_chunks_per_second": self.throughput_chunks_per_second,
            "success_rate": self.success_rate,
            "memory_usage_mb": self.memory_usage_mb
        }


class PerformanceBenchmarker:
    """Performance benchmarking utility for pipeline testing."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
    
    async def benchmark_pipeline_latency(
        self, 
        pipeline: VoicePipeline, 
        num_samples: int = 100,
        audio_duration_seconds: float = 2.0
    ) -> BenchmarkResult:
        """Benchmark end-to-end pipeline latency."""
        latencies = []
        successes = 0
        failures = 0
        
        # Monitor memory usage
        import psutil
        import os
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        start_benchmark_time = time.time()
        
        for i in range(num_samples):
            # Generate synthetic audio
            audio_data = SyntheticAudioGenerator.generate_speech_audio(
                duration_seconds=audio_duration_seconds
            )
            audio_chunk = AudioChunk(
                data=audio_data,
                timestamp=time.time(),
                sample_rate=self.config.sample_rate,
                duration_ms=audio_duration_seconds * 1000
            )
            
            # Measure processing latency
            start_time = time.time()
            try:
                result = await pipeline.process_audio_chunk(audio_chunk)
                end_time = time.time()
                
                if result is not None:
                    latency_ms = (end_time - start_time) * 1000
                    latencies.append(latency_ms)
                    successes += 1
                else:
                    failures += 1
            except Exception as e:
                failures += 1
                print(f"Processing failed for sample {i}: {e}")
        
        end_benchmark_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate statistics
        if latencies:
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
        else:
            avg_latency = min_latency = max_latency = p95_latency = p99_latency = 0
        
        total_time = end_benchmark_time - start_benchmark_time
        throughput = num_samples / total_time if total_time > 0 else 0
        success_rate = successes / num_samples if num_samples > 0 else 0
        memory_usage = final_memory - initial_memory
        
        result = BenchmarkResult(
            test_name="pipeline_latency",
            num_samples=num_samples,
            avg_latency_ms=avg_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_chunks_per_second=throughput,
            success_rate=success_rate,
            memory_usage_mb=memory_usage
        )
        
        self.results.append(result)
        return result
    
    async def benchmark_component_latency(
        self, 
        pipeline: VoicePipeline, 
        num_samples: int = 50
    ) -> Dict[str, BenchmarkResult]:
        """Benchmark individual component latencies."""
        component_results = {}
        
        # Test STT component
        stt_latencies = []
        for i in range(num_samples):
            audio_data = SyntheticAudioGenerator.generate_speech_audio()
            audio_chunk = AudioChunk(
                data=audio_data,
                timestamp=time.time(),
                sample_rate=self.config.sample_rate,
                duration_ms=len(audio_data) / self.config.sample_rate * 1000
            )
            
            start_time = time.time()
            try:
                # Access STT processor directly for component testing
                if hasattr(pipeline, '_stt_processor'):
                    result = await pipeline._stt_processor.process_audio(audio_chunk)
                    end_time = time.time()
                    if result:
                        stt_latencies.append((end_time - start_time) * 1000)
            except Exception:
                pass
        
        if stt_latencies:
            component_results['stt'] = BenchmarkResult(
                test_name="stt_component",
                num_samples=len(stt_latencies),
                avg_latency_ms=statistics.mean(stt_latencies),
                min_latency_ms=min(stt_latencies),
                max_latency_ms=max(stt_latencies),
                p95_latency_ms=np.percentile(stt_latencies, 95),
                p99_latency_ms=np.percentile(stt_latencies, 99),
                throughput_chunks_per_second=len(stt_latencies) / sum(stt_latencies) * 1000,
                success_rate=1.0,
                memory_usage_mb=0
            )
        
        return component_results
    
    async def benchmark_concurrent_load(
        self, 
        pipeline: VoicePipeline, 
        num_concurrent: int = 10,
        num_samples_per_worker: int = 20
    ) -> BenchmarkResult:
        """Benchmark pipeline under concurrent load."""
        
        async def worker_task(worker_id: int) -> List[float]:
            """Worker task for concurrent processing."""
            worker_latencies = []
            
            for i in range(num_samples_per_worker):
                audio_data = SyntheticAudioGenerator.generate_speech_audio()
                audio_chunk = AudioChunk(
                    data=audio_data,
                    timestamp=time.time(),
                    sample_rate=self.config.sample_rate,
                    duration_ms=len(audio_data) / self.config.sample_rate * 1000
                )
                
                start_time = time.time()
                try:
                    result = await pipeline.process_audio_chunk(audio_chunk)
                    end_time = time.time()
                    
                    if result is not None:
                        latency_ms = (end_time - start_time) * 1000
                        worker_latencies.append(latency_ms)
                except Exception:
                    pass
            
            return worker_latencies
        
        # Monitor memory usage
        import psutil
        import os
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run concurrent workers
        start_time = time.time()
        tasks = [worker_task(i) for i in range(num_concurrent)]
        worker_results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Aggregate results
        all_latencies = []
        for worker_latencies in worker_results:
            all_latencies.extend(worker_latencies)
        
        total_samples = num_concurrent * num_samples_per_worker
        successful_samples = len(all_latencies)
        
        if all_latencies:
            avg_latency = statistics.mean(all_latencies)
            min_latency = min(all_latencies)
            max_latency = max(all_latencies)
            p95_latency = np.percentile(all_latencies, 95)
            p99_latency = np.percentile(all_latencies, 99)
        else:
            avg_latency = min_latency = max_latency = p95_latency = p99_latency = 0
        
        total_time = end_time - start_time
        throughput = successful_samples / total_time if total_time > 0 else 0
        success_rate = successful_samples / total_samples if total_samples > 0 else 0
        memory_usage = final_memory - initial_memory
        
        result = BenchmarkResult(
            test_name=f"concurrent_load_{num_concurrent}_workers",
            num_samples=total_samples,
            avg_latency_ms=avg_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_chunks_per_second=throughput,
            success_rate=success_rate,
            memory_usage_mb=memory_usage
        )
        
        self.results.append(result)
        return result
    
    def save_results(self, output_file: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        results_data = {
            "timestamp": time.time(),
            "config": {
                "sample_rate": self.config.sample_rate,
                "chunk_size": self.config.chunk_size,
                "max_latency_ms": self.config.max_latency_ms,
                "gpu_device": self.config.gpu_device
            },
            "results": [result.to_dict() for result in self.results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"Benchmark results saved to {output_file}")
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*60)
        
        for result in self.results:
            print(f"\nTest: {result.test_name}")
            print(f"  Samples: {result.num_samples}")
            print(f"  Average Latency: {result.avg_latency_ms:.2f}ms")
            print(f"  95th Percentile: {result.p95_latency_ms:.2f}ms")
            print(f"  99th Percentile: {result.p99_latency_ms:.2f}ms")
            print(f"  Throughput: {result.throughput_chunks_per_second:.2f} chunks/sec")
            print(f"  Success Rate: {result.success_rate:.2%}")
            print(f"  Memory Usage: {result.memory_usage_mb:.2f}MB")


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.fixture
    def benchmark_config(self):
        """Configuration optimized for benchmarking."""
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
    def mock_components(self):
        """Mock components with realistic timing."""
        components = MockComponents()
        
        # Add realistic delays to mock components
        original_stt = components.get_mock_stt_processor()
        original_stt.process_audio = AsyncMock(side_effect=lambda x: asyncio.sleep(0.1).then(lambda: "test text"))
        
        return components
    
    @pytest.mark.asyncio
    async def test_baseline_latency_benchmark(self, benchmark_config, mock_components):
        """Benchmark baseline pipeline latency performance."""
        with patch('src.core.pipeline.STTProcessor', return_value=mock_components.get_mock_stt_processor()), \
             patch('src.core.pipeline.CharacterTransformer', return_value=mock_components.get_mock_character_transformer()), \
             patch('src.core.pipeline.TTSProcessor', return_value=mock_components.get_mock_tts_processor()), \
             patch('src.core.pipeline.AudioCapture'), \
             patch('src.core.pipeline.AudioOutput'):
            
            pipeline = VoicePipeline(benchmark_config)
            await pipeline.initialize()
            await pipeline.start()
            
            benchmarker = PerformanceBenchmarker(benchmark_config)
            
            # Run baseline latency benchmark
            result = await benchmarker.benchmark_pipeline_latency(
                pipeline, 
                num_samples=50,
                audio_duration_seconds=2.0
            )
            
            # Verify performance requirements
            assert result.avg_latency_ms < benchmark_config.max_latency_ms
            assert result.p95_latency_ms < benchmark_config.max_latency_ms * 1.2
            assert result.success_rate > 0.95
            
            await pipeline.stop()
            
            # Print results for analysis
            benchmarker.print_summary()
    
    @pytest.mark.asyncio
    async def test_character_switching_performance(self, benchmark_config, mock_components):
        """Benchmark performance when switching between characters."""
        with patch('src.core.pipeline.STTProcessor', return_value=mock_components.get_mock_stt_processor()), \
             patch('src.core.pipeline.CharacterTransformer', return_value=mock_components.get_mock_character_transformer()), \
             patch('src.core.pipeline.TTSProcessor', return_value=mock_components.get_mock_tts_processor()), \
             patch('src.core.pipeline.AudioCapture'), \
             patch('src.core.pipeline.AudioOutput'):
            
            pipeline = VoicePipeline(benchmark_config)
            await pipeline.initialize()
            await pipeline.start()
            
            characters = ["anime-waifu", "patriotic-american", "slurring-drunk"]
            character_switch_times = []
            
            for character in characters:
                start_time = time.time()
                await pipeline.set_character(character)
                end_time = time.time()
                
                switch_time_ms = (end_time - start_time) * 1000
                character_switch_times.append(switch_time_ms)
                
                # Process a few chunks with this character
                for _ in range(5):
                    audio_data = SyntheticAudioGenerator.generate_speech_audio()
                    audio_chunk = AudioChunk(
                        data=audio_data,
                        timestamp=time.time(),
                        sample_rate=benchmark_config.sample_rate,
                        duration_ms=len(audio_data) / benchmark_config.sample_rate * 1000
                    )
                    
                    await pipeline.process_audio_chunk(audio_chunk)
            
            # Verify character switching performance
            avg_switch_time = statistics.mean(character_switch_times)
            max_switch_time = max(character_switch_times)
            
            # Character switching should be fast (under 5 seconds as per requirements)
            assert avg_switch_time < 5000, f"Average character switch time: {avg_switch_time}ms"
            assert max_switch_time < 5000, f"Max character switch time: {max_switch_time}ms"
            
            await pipeline.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_load_benchmark(self, benchmark_config, mock_components):
        """Benchmark pipeline performance under concurrent load."""
        with patch('src.core.pipeline.STTProcessor', return_value=mock_components.get_mock_stt_processor()), \
             patch('src.core.pipeline.CharacterTransformer', return_value=mock_components.get_mock_character_transformer()), \
             patch('src.core.pipeline.TTSProcessor', return_value=mock_components.get_mock_tts_processor()), \
             patch('src.core.pipeline.AudioCapture'), \
             patch('src.core.pipeline.AudioOutput'):
            
            pipeline = VoicePipeline(benchmark_config)
            await pipeline.initialize()
            await pipeline.start()
            
            benchmarker = PerformanceBenchmarker(benchmark_config)
            
            # Test different concurrent loads
            concurrent_loads = [1, 3, 5, 10]
            
            for num_concurrent in concurrent_loads:
                result = await benchmarker.benchmark_concurrent_load(
                    pipeline,
                    num_concurrent=num_concurrent,
                    num_samples_per_worker=10
                )
                
                # Verify that concurrent processing doesn't degrade too much
                # Allow some degradation but should still meet basic requirements
                max_acceptable_latency = benchmark_config.max_latency_ms * (1 + num_concurrent * 0.1)
                assert result.avg_latency_ms < max_acceptable_latency
                assert result.success_rate > 0.8  # Allow some failures under high load
            
            await pipeline.stop()
            
            benchmarker.print_summary()
    
    @pytest.mark.asyncio
    async def test_memory_usage_benchmark(self, benchmark_config, mock_components):
        """Benchmark memory usage during extended operation."""
        with patch('src.core.pipeline.STTProcessor', return_value=mock_components.get_mock_stt_processor()), \
             patch('src.core.pipeline.CharacterTransformer', return_value=mock_components.get_mock_character_transformer()), \
             patch('src.core.pipeline.TTSProcessor', return_value=mock_components.get_mock_tts_processor()), \
             patch('src.core.pipeline.AudioCapture'), \
             patch('src.core.pipeline.AudioOutput'):
            
            pipeline = VoicePipeline(benchmark_config)
            await pipeline.initialize()
            await pipeline.start()
            
            import psutil
            import os
            process = psutil.Process(os.getpid())
            
            # Monitor memory usage over time
            memory_samples = []
            num_processing_cycles = 20
            samples_per_cycle = 10
            
            for cycle in range(num_processing_cycles):
                # Process audio chunks
                for i in range(samples_per_cycle):
                    audio_data = SyntheticAudioGenerator.generate_speech_audio()
                    audio_chunk = AudioChunk(
                        data=audio_data,
                        timestamp=time.time(),
                        sample_rate=benchmark_config.sample_rate,
                        duration_ms=len(audio_data) / benchmark_config.sample_rate * 1000
                    )
                    
                    await pipeline.process_audio_chunk(audio_chunk)
                
                # Sample memory usage
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_samples.append(memory_mb)
                
                # Small delay between cycles
                await asyncio.sleep(0.1)
            
            # Analyze memory usage
            initial_memory = memory_samples[0]
            final_memory = memory_samples[-1]
            max_memory = max(memory_samples)
            memory_growth = final_memory - initial_memory
            
            # Memory growth should be reasonable (less than 50MB for testing)
            assert memory_growth < 50, f"Memory grew by {memory_growth}MB"
            
            # Peak memory usage should be reasonable
            assert max_memory < initial_memory + 100, f"Peak memory usage: {max_memory}MB"
            
            await pipeline.stop()
    
    @pytest.mark.asyncio
    async def test_audio_quality_benchmark(self, benchmark_config, mock_components):
        """Benchmark audio processing quality metrics."""
        with patch('src.core.pipeline.STTProcessor', return_value=mock_components.get_mock_stt_processor()), \
             patch('src.core.pipeline.CharacterTransformer', return_value=mock_components.get_mock_character_transformer()), \
             patch('src.core.pipeline.TTSProcessor', return_value=mock_components.get_mock_tts_processor()), \
             patch('src.core.pipeline.AudioCapture'), \
             patch('src.core.pipeline.AudioOutput'):
            
            pipeline = VoicePipeline(benchmark_config)
            await pipeline.initialize()
            await pipeline.start()
            
            # Test different audio qualities
            audio_durations = [0.5, 1.0, 2.0, 3.0, 5.0]  # seconds
            quality_results = {}
            
            for duration in audio_durations:
                latencies = []
                
                for _ in range(10):  # Test each duration multiple times
                    audio_data = SyntheticAudioGenerator.generate_speech_audio(
                        duration_seconds=duration
                    )
                    audio_chunk = AudioChunk(
                        data=audio_data,
                        timestamp=time.time(),
                        sample_rate=benchmark_config.sample_rate,
                        duration_ms=duration * 1000
                    )
                    
                    start_time = time.time()
                    result = await pipeline.process_audio_chunk(audio_chunk)
                    end_time = time.time()
                    
                    if result is not None:
                        latency_ms = (end_time - start_time) * 1000
                        latencies.append(latency_ms)
                
                if latencies:
                    quality_results[duration] = {
                        'avg_latency_ms': statistics.mean(latencies),
                        'max_latency_ms': max(latencies),
                        'success_rate': len(latencies) / 10
                    }
            
            # Verify that longer audio doesn't cause exponential latency growth
            for duration, metrics in quality_results.items():
                # Latency should scale reasonably with audio duration
                expected_max_latency = benchmark_config.max_latency_ms + (duration * 500)  # Allow 500ms per second
                assert metrics['avg_latency_ms'] < expected_max_latency
                assert metrics['success_rate'] > 0.8
            
            await pipeline.stop()


if __name__ == "__main__":
    # Run benchmark tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "benchmark"])