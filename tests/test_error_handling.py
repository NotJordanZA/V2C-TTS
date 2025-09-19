"""
Unit tests for the error handling and recovery system.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from src.core.error_handling import (
    ErrorSeverity, RecoveryStrategy, ErrorContext, RetryConfig, ErrorRule,
    ErrorRecoveryManager, GracefulDegradationManager
)
from src.core.interfaces import PipelineStage


class TestErrorContext:
    """Test error context functionality."""
    
    def test_error_context_creation(self):
        """Test error context is created correctly."""
        error = ValueError("Test error")
        context = ErrorContext(
            error=error,
            stage=PipelineStage.SPEECH_TO_TEXT,
            timestamp=datetime.now(),
            severity=ErrorSeverity.HIGH,
            metadata={"key": "value"}
        )
        
        assert context.error == error
        assert context.stage == PipelineStage.SPEECH_TO_TEXT
        assert context.severity == ErrorSeverity.HIGH
        assert context.metadata == {"key": "value"}
        assert context.stack_trace is not None
    
    def test_error_context_auto_stack_trace(self):
        """Test error context automatically captures stack trace."""
        try:
            raise ValueError("Test error")
        except ValueError as e:
            context = ErrorContext(
                error=e,
                stage=PipelineStage.SPEECH_TO_TEXT,
                timestamp=datetime.now(),
                severity=ErrorSeverity.HIGH
            )
            
            assert "ValueError: Test error" in context.stack_trace


class TestRetryConfig:
    """Test retry configuration functionality."""
    
    def test_retry_config_defaults(self):
        """Test retry config default values."""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
    
    def test_get_delay_exponential(self):
        """Test exponential delay calculation."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=False)
        
        assert config.get_delay(0) == 1.0  # 1.0 * 2^0
        assert config.get_delay(1) == 2.0  # 1.0 * 2^1
        assert config.get_delay(2) == 4.0  # 1.0 * 2^2
    
    def test_get_delay_max_limit(self):
        """Test delay respects maximum limit."""
        config = RetryConfig(base_delay=10.0, max_delay=15.0, exponential_base=2.0, jitter=False)
        
        assert config.get_delay(0) == 10.0
        assert config.get_delay(1) == 15.0  # Capped at max_delay
        assert config.get_delay(2) == 15.0  # Still capped
    
    def test_get_delay_with_jitter(self):
        """Test delay with jitter applied."""
        config = RetryConfig(base_delay=10.0, jitter=True)
        
        delay = config.get_delay(0)
        # With jitter, delay should be between 5.0 and 10.0
        assert 5.0 <= delay <= 10.0


class TestErrorRule:
    """Test error rule functionality."""
    
    def test_error_rule_creation(self):
        """Test error rule is created correctly."""
        rule = ErrorRule(
            error_types=[ValueError, TypeError],
            stages=[PipelineStage.SPEECH_TO_TEXT],
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.RETRY
        )
        
        assert rule.error_types == [ValueError, TypeError]
        assert rule.stages == [PipelineStage.SPEECH_TO_TEXT]
        assert rule.severity == ErrorSeverity.HIGH
        assert rule.recovery_strategy == RecoveryStrategy.RETRY
    
    def test_error_rule_matches_error_type(self):
        """Test error rule matches correct error types."""
        rule = ErrorRule(
            error_types=[ValueError, TypeError],
            stages=[PipelineStage.SPEECH_TO_TEXT],
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.RETRY
        )
        
        assert rule.matches(ValueError("test"), PipelineStage.SPEECH_TO_TEXT)
        assert rule.matches(TypeError("test"), PipelineStage.SPEECH_TO_TEXT)
        assert not rule.matches(RuntimeError("test"), PipelineStage.SPEECH_TO_TEXT)
    
    def test_error_rule_matches_stage(self):
        """Test error rule matches correct stages."""
        rule = ErrorRule(
            error_types=[ValueError],
            stages=[PipelineStage.SPEECH_TO_TEXT, PipelineStage.TEXT_TO_SPEECH],
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.RETRY
        )
        
        assert rule.matches(ValueError("test"), PipelineStage.SPEECH_TO_TEXT)
        assert rule.matches(ValueError("test"), PipelineStage.TEXT_TO_SPEECH)
        assert not rule.matches(ValueError("test"), PipelineStage.AUDIO_CAPTURE)
    
    def test_error_rule_matches_any_stage(self):
        """Test error rule matches any stage when stages list is empty."""
        rule = ErrorRule(
            error_types=[ValueError],
            stages=[],  # Empty list means match any stage
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.RETRY
        )
        
        assert rule.matches(ValueError("test"), PipelineStage.SPEECH_TO_TEXT)
        assert rule.matches(ValueError("test"), PipelineStage.AUDIO_CAPTURE)
        assert rule.matches(ValueError("test"), PipelineStage.TEXT_TO_SPEECH)


class TestErrorRecoveryManager:
    """Test error recovery manager functionality."""
    
    def test_manager_initialization(self):
        """Test manager is initialized correctly."""
        manager = ErrorRecoveryManager()
        
        assert len(manager.error_rules) > 0  # Should have default rules
        assert len(manager.error_history) == 0
        assert len(manager.retry_counts) == 0
        assert manager.max_history_size == 1000
    
    def test_add_error_rule(self):
        """Test adding custom error rule."""
        manager = ErrorRecoveryManager()
        initial_rule_count = len(manager.error_rules)
        
        rule = ErrorRule(
            error_types=[ValueError],
            stages=[PipelineStage.SPEECH_TO_TEXT],
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.RETRY
        )
        
        manager.add_error_rule(rule)
        
        assert len(manager.error_rules) == initial_rule_count + 1
        assert rule in manager.error_rules
    
    def test_remove_error_rule(self):
        """Test removing error rule."""
        manager = ErrorRecoveryManager()
        
        rule = ErrorRule(
            error_types=[ValueError],
            stages=[PipelineStage.SPEECH_TO_TEXT],
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.RETRY
        )
        
        manager.add_error_rule(rule)
        assert rule in manager.error_rules
        
        manager.remove_error_rule(rule)
        assert rule not in manager.error_rules
    
    @pytest.mark.asyncio
    async def test_handle_error_with_matching_rule(self):
        """Test handling error with matching rule."""
        manager = ErrorRecoveryManager()
        
        # Add a specific rule for testing
        rule = ErrorRule(
            error_types=[ValueError],
            stages=[PipelineStage.SPEECH_TO_TEXT],
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.SKIP  # Simple strategy for testing
        )
        manager.error_rules.insert(0, rule)  # Insert at beginning to match first
        
        error = ValueError("Test error")
        result = await manager.handle_error(error, PipelineStage.SPEECH_TO_TEXT)
        
        assert result is True  # Skip strategy should return True
        assert len(manager.error_history) == 1
        assert manager.error_history[0].error == error
        assert manager.error_history[0].severity == ErrorSeverity.HIGH
    
    @pytest.mark.asyncio
    async def test_handle_error_no_matching_rule(self):
        """Test handling error with no matching rule."""
        manager = ErrorRecoveryManager()
        manager.error_rules.clear()  # Remove all rules
        
        error = ValueError("Test error")
        result = await manager.handle_error(error, PipelineStage.SPEECH_TO_TEXT)
        
        assert result is False  # Should fail when no rule matches
        assert len(manager.error_history) == 1
    
    @pytest.mark.asyncio
    async def test_retry_strategy_success(self):
        """Test retry strategy within limits."""
        manager = ErrorRecoveryManager()
        
        rule = ErrorRule(
            error_types=[ValueError],
            stages=[PipelineStage.SPEECH_TO_TEXT],
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.RETRY,
            retry_config=RetryConfig(max_attempts=3, base_delay=0.01)  # Fast for testing
        )
        manager.error_rules.insert(0, rule)
        
        error = ValueError("Test error")
        
        # First retry should succeed
        result = await manager.handle_error(error, PipelineStage.SPEECH_TO_TEXT)
        assert result is True
        
        # Check retry count was incremented
        retry_key = "speech_to_text_ValueError"
        assert manager.retry_counts[retry_key] == 1
    
    @pytest.mark.asyncio
    async def test_retry_strategy_exhausted(self):
        """Test retry strategy when attempts are exhausted."""
        manager = ErrorRecoveryManager()
        
        rule = ErrorRule(
            error_types=[ValueError],
            stages=[PipelineStage.SPEECH_TO_TEXT],
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.RETRY,
            retry_config=RetryConfig(max_attempts=2, base_delay=0.01)
        )
        manager.error_rules.insert(0, rule)
        
        error = ValueError("Test error")
        retry_key = "speech_to_text_ValueError"
        
        # Simulate previous attempts
        manager.retry_counts[retry_key] = 2  # Already at max attempts
        
        result = await manager.handle_error(error, PipelineStage.SPEECH_TO_TEXT)
        assert result is False  # Should fail when max attempts reached
        assert manager.retry_counts[retry_key] == 0  # Should reset
    
    @pytest.mark.asyncio
    async def test_fallback_strategy_with_action(self):
        """Test fallback strategy with custom action."""
        manager = ErrorRecoveryManager()
        
        async def mock_fallback_action(context):
            context.metadata['fallback_executed'] = True
            return True
        
        rule = ErrorRule(
            error_types=[ValueError],
            stages=[PipelineStage.SPEECH_TO_TEXT],
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.FALLBACK,
            fallback_action=mock_fallback_action
        )
        manager.error_rules.insert(0, rule)
        
        error = ValueError("Test error")
        result = await manager.handle_error(error, PipelineStage.SPEECH_TO_TEXT)
        
        assert result is True
        assert manager.error_history[0].metadata.get('fallback_executed') is True
    
    @pytest.mark.asyncio
    async def test_fallback_strategy_no_action(self):
        """Test fallback strategy without custom action."""
        manager = ErrorRecoveryManager()
        
        rule = ErrorRule(
            error_types=[ValueError],
            stages=[PipelineStage.SPEECH_TO_TEXT],
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.FALLBACK,
            fallback_action=None
        )
        manager.error_rules.insert(0, rule)
        
        error = ValueError("Test error")
        result = await manager.handle_error(error, PipelineStage.SPEECH_TO_TEXT)
        
        assert result is False  # Should fail without fallback action
    
    @pytest.mark.asyncio
    async def test_restart_component_strategy(self):
        """Test restart component strategy."""
        manager = ErrorRecoveryManager()
        
        rule = ErrorRule(
            error_types=[MemoryError],
            stages=[PipelineStage.SPEECH_TO_TEXT],
            severity=ErrorSeverity.CRITICAL,
            recovery_strategy=RecoveryStrategy.RESTART_COMPONENT
        )
        manager.error_rules.insert(0, rule)
        
        error = MemoryError("Out of memory")
        result = await manager.handle_error(error, PipelineStage.SPEECH_TO_TEXT)
        
        assert result is True
        assert manager.component_restart_counts["speech_to_text"] == 1
    
    @pytest.mark.asyncio
    async def test_restart_component_limit_exceeded(self):
        """Test restart component strategy when limit is exceeded."""
        manager = ErrorRecoveryManager()
        manager.max_component_restarts = 2
        
        rule = ErrorRule(
            error_types=[MemoryError],
            stages=[PipelineStage.SPEECH_TO_TEXT],
            severity=ErrorSeverity.CRITICAL,
            recovery_strategy=RecoveryStrategy.RESTART_COMPONENT
        )
        manager.error_rules.insert(0, rule)
        
        # Simulate previous restarts
        manager.component_restart_counts["speech_to_text"] = 2
        
        error = MemoryError("Out of memory")
        result = await manager.handle_error(error, PipelineStage.SPEECH_TO_TEXT)
        
        assert result is False  # Should fail when restart limit exceeded
    
    def test_get_error_statistics_empty(self):
        """Test getting error statistics when no errors."""
        manager = ErrorRecoveryManager()
        
        stats = manager.get_error_statistics()
        
        assert stats['total_errors'] == 0
        assert stats['error_rate'] == 0.0
        assert stats['errors_by_stage'] == {}
        assert stats['errors_by_severity'] == {}
        assert stats['most_common_errors'] == []
    
    def test_get_error_statistics_with_data(self):
        """Test getting error statistics with error data."""
        manager = ErrorRecoveryManager()
        
        # Add some test errors
        errors = [
            ErrorContext(
                error=ValueError("Error 1"),
                stage=PipelineStage.SPEECH_TO_TEXT,
                timestamp=datetime.now(),
                severity=ErrorSeverity.HIGH
            ),
            ErrorContext(
                error=ValueError("Error 2"),
                stage=PipelineStage.SPEECH_TO_TEXT,
                timestamp=datetime.now(),
                severity=ErrorSeverity.MEDIUM
            ),
            ErrorContext(
                error=RuntimeError("Error 3"),
                stage=PipelineStage.TEXT_TO_SPEECH,
                timestamp=datetime.now(),
                severity=ErrorSeverity.HIGH
            )
        ]
        
        manager.error_history.extend(errors)
        
        stats = manager.get_error_statistics()
        
        assert stats['total_errors'] == 3
        assert stats['errors_by_stage']['speech_to_text'] == 2
        assert stats['errors_by_stage']['text_to_speech'] == 1
        assert stats['errors_by_severity']['high'] == 2
        assert stats['errors_by_severity']['medium'] == 1
        assert ('ValueError', 2) in stats['most_common_errors']
        assert ('RuntimeError', 1) in stats['most_common_errors']
    
    def test_clear_error_history(self):
        """Test clearing error history."""
        manager = ErrorRecoveryManager()
        
        # Add some test data
        manager.error_history.append(ErrorContext(
            error=ValueError("Test"),
            stage=PipelineStage.SPEECH_TO_TEXT,
            timestamp=datetime.now(),
            severity=ErrorSeverity.HIGH
        ))
        manager.retry_counts["test"] = 1
        manager.component_restart_counts["test"] = 1
        
        manager.clear_error_history()
        
        assert len(manager.error_history) == 0
        assert len(manager.retry_counts) == 0
        assert len(manager.component_restart_counts) == 0
    
    def test_get_recent_errors(self):
        """Test getting recent errors."""
        manager = ErrorRecoveryManager()
        
        # Add test errors
        for i in range(15):
            manager.error_history.append(ErrorContext(
                error=ValueError(f"Error {i}"),
                stage=PipelineStage.SPEECH_TO_TEXT,
                timestamp=datetime.now(),
                severity=ErrorSeverity.HIGH
            ))
        
        recent_errors = manager.get_recent_errors(5)
        
        assert len(recent_errors) == 5
        # Should get the last 5 errors
        assert str(recent_errors[0].error) == "Error 10"
        assert str(recent_errors[4].error) == "Error 14"


class TestGracefulDegradationManager:
    """Test graceful degradation manager functionality."""
    
    def test_manager_initialization(self):
        """Test manager is initialized correctly."""
        manager = GracefulDegradationManager()
        
        assert len(manager.degradation_state) == 0
        assert len(manager.fallback_configs) > 0  # Should have default configs
        assert 'gpu_to_cpu' in manager.fallback_configs
        assert 'quality_degradation' in manager.fallback_configs
        assert 'performance_degradation' in manager.fallback_configs
    
    def test_enable_degradation(self):
        """Test enabling degradation."""
        manager = GracefulDegradationManager()
        
        manager.enable_degradation('gpu_to_cpu', 'GPU unavailable')
        
        assert manager.is_degraded('gpu_to_cpu')
        assert manager.degradation_state['gpu_to_cpu']['enabled'] is True
        assert manager.degradation_state['gpu_to_cpu']['reason'] == 'GPU unavailable'
    
    def test_disable_degradation(self):
        """Test disabling degradation."""
        manager = GracefulDegradationManager()
        
        manager.enable_degradation('gpu_to_cpu', 'GPU unavailable')
        assert manager.is_degraded('gpu_to_cpu')
        
        manager.disable_degradation('gpu_to_cpu')
        assert not manager.is_degraded('gpu_to_cpu')
    
    def test_enable_unknown_degradation(self):
        """Test enabling unknown degradation type."""
        manager = GracefulDegradationManager()
        
        # Should not raise error, just log
        manager.enable_degradation('unknown_type', 'Test reason')
        
        assert not manager.is_degraded('unknown_type')
    
    def test_get_degraded_config(self):
        """Test getting configuration with degradation applied."""
        manager = GracefulDegradationManager()
        
        base_config = {
            'stt_model_size': 'large',
            'audio_sample_rate': 44100,
            'other_setting': 'value'
        }
        
        # Enable degradation
        manager.enable_degradation('gpu_to_cpu', 'GPU unavailable')
        
        degraded_config = manager.get_degraded_config(base_config)
        
        # Should have fallback values applied
        assert degraded_config['stt_model_size'] == 'tiny'  # From gpu_to_cpu fallback
        assert degraded_config['other_setting'] == 'value'  # Original value preserved
    
    def test_get_degraded_config_multiple_degradations(self):
        """Test getting configuration with multiple degradations."""
        manager = GracefulDegradationManager()
        
        base_config = {
            'stt_model_size': 'large',
            'audio_sample_rate': 44100
        }
        
        # Enable multiple degradations
        manager.enable_degradation('gpu_to_cpu', 'GPU unavailable')
        manager.enable_degradation('quality_degradation', 'Performance issues')
        
        degraded_config = manager.get_degraded_config(base_config)
        
        # Later degradation should override earlier ones
        assert degraded_config['stt_model_size'] == 'base'  # From quality_degradation
        assert degraded_config['audio_sample_rate'] == 16000  # From quality_degradation
    
    def test_get_degradation_status(self):
        """Test getting degradation status."""
        manager = GracefulDegradationManager()
        
        manager.enable_degradation('gpu_to_cpu', 'GPU unavailable')
        
        status = manager.get_degradation_status()
        
        assert 'gpu_to_cpu' in status['active_degradations']
        assert 'gpu_to_cpu' in status['degradation_details']
        assert len(status['available_degradations']) >= 3  # At least the default ones


if __name__ == "__main__":
    pytest.main([__file__])