"""
Comprehensive error handling and recovery system for the voice transformation pipeline.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Callable, Any, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import traceback

from .interfaces import PipelineError, PipelineStage


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(str, Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    RESTART_COMPONENT = "restart_component"
    RESTART_PIPELINE = "restart_pipeline"
    FAIL = "fail"


@dataclass
class ErrorContext:
    """Context information for an error occurrence."""
    error: Exception
    stage: PipelineStage
    timestamp: datetime
    severity: ErrorSeverity
    metadata: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    
    def __post_init__(self):
        if self.stack_trace is None:
            self.stack_trace = traceback.format_exc()


@dataclass
class RetryConfig:
    """Configuration for retry logic."""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = min(self.base_delay * (self.exponential_base ** attempt), self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay


@dataclass
class ErrorRule:
    """Rule for handling specific types of errors."""
    error_types: List[Type[Exception]]
    stages: List[PipelineStage]
    severity: ErrorSeverity
    recovery_strategy: RecoveryStrategy
    retry_config: Optional[RetryConfig] = None
    fallback_action: Optional[Callable] = None
    custom_handler: Optional[Callable] = None
    
    def matches(self, error: Exception, stage: PipelineStage) -> bool:
        """Check if this rule matches the given error and stage."""
        error_match = any(isinstance(error, error_type) for error_type in self.error_types)
        stage_match = not self.stages or stage in self.stages
        return error_match and stage_match


class ErrorRecoveryManager:
    """Manages error handling and recovery strategies."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_rules: List[ErrorRule] = []
        self.error_history: List[ErrorContext] = []
        self.retry_counts: Dict[str, int] = {}
        self.component_restart_counts: Dict[str, int] = {}
        self.max_history_size = 1000
        
        # Default retry configuration
        self.default_retry_config = RetryConfig()
        
        # Component restart limits
        self.max_component_restarts = 3
        self.component_restart_window = timedelta(minutes=10)
        
        # Setup default error rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default error handling rules."""
        
        # GPU/CUDA errors - fallback to CPU
        self.add_error_rule(ErrorRule(
            error_types=[RuntimeError],  # CUDA errors often manifest as RuntimeError
            stages=[PipelineStage.SPEECH_TO_TEXT, PipelineStage.CHARACTER_TRANSFORM, PipelineStage.TEXT_TO_SPEECH],
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.FALLBACK,
            fallback_action=self._fallback_to_cpu
        ))
        
        # Memory errors - restart component
        self.add_error_rule(ErrorRule(
            error_types=[MemoryError, OSError],
            stages=list(PipelineStage),
            severity=ErrorSeverity.CRITICAL,
            recovery_strategy=RecoveryStrategy.RESTART_COMPONENT,
            retry_config=RetryConfig(max_attempts=2, base_delay=5.0)
        ))
        
        # Network/IO errors - retry with backoff
        self.add_error_rule(ErrorRule(
            error_types=[ConnectionError, TimeoutError, IOError],
            stages=list(PipelineStage),
            severity=ErrorSeverity.MEDIUM,
            recovery_strategy=RecoveryStrategy.RETRY,
            retry_config=RetryConfig(max_attempts=5, base_delay=2.0)
        ))
        
        # Audio device errors - retry with different device
        self.add_error_rule(ErrorRule(
            error_types=[OSError, ValueError],  # PyAudio errors
            stages=[PipelineStage.AUDIO_CAPTURE, PipelineStage.AUDIO_OUTPUT],
            severity=ErrorSeverity.HIGH,
            recovery_strategy=RecoveryStrategy.FALLBACK,
            fallback_action=self._fallback_audio_device
        ))
        
        # Model loading errors - retry then fail
        self.add_error_rule(ErrorRule(
            error_types=[FileNotFoundError, ImportError, ModuleNotFoundError],
            stages=list(PipelineStage),
            severity=ErrorSeverity.CRITICAL,
            recovery_strategy=RecoveryStrategy.RETRY,
            retry_config=RetryConfig(max_attempts=2, base_delay=1.0)
        ))
        
        # Generic exceptions - retry with exponential backoff
        self.add_error_rule(ErrorRule(
            error_types=[Exception],
            stages=list(PipelineStage),
            severity=ErrorSeverity.LOW,
            recovery_strategy=RecoveryStrategy.RETRY,
            retry_config=RetryConfig(max_attempts=3, base_delay=1.0)
        ))
    
    def add_error_rule(self, rule: ErrorRule):
        """Add a new error handling rule."""
        self.error_rules.append(rule)
    
    def remove_error_rule(self, rule: ErrorRule):
        """Remove an error handling rule."""
        if rule in self.error_rules:
            self.error_rules.remove(rule)
    
    async def handle_error(self, error: Exception, stage: PipelineStage, 
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Handle an error according to configured rules.
        
        Returns:
            bool: True if error was handled and operation should continue, False if it should fail
        """
        # Create error context
        context = ErrorContext(
            error=error,
            stage=stage,
            timestamp=datetime.now(),
            severity=ErrorSeverity.LOW,  # Will be updated by matching rule
            metadata=metadata or {}
        )
        
        # Find matching rule
        matching_rule = self._find_matching_rule(error, stage)
        if matching_rule:
            context.severity = matching_rule.severity
            
            self.logger.error(
                f"Error in {stage.value}: {error} (Severity: {context.severity.value})"
            )
            
            # Record error in history
            self._record_error(context)
            
            # Execute recovery strategy
            return await self._execute_recovery_strategy(matching_rule, context)
        else:
            # No matching rule - log and fail
            self.logger.error(f"Unhandled error in {stage.value}: {error}")
            self._record_error(context)
            return False
    
    def _find_matching_rule(self, error: Exception, stage: PipelineStage) -> Optional[ErrorRule]:
        """Find the first matching error rule."""
        for rule in self.error_rules:
            if rule.matches(error, stage):
                return rule
        return None
    
    def _record_error(self, context: ErrorContext):
        """Record error in history."""
        self.error_history.append(context)
        
        # Trim history if too large
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    async def _execute_recovery_strategy(self, rule: ErrorRule, context: ErrorContext) -> bool:
        """Execute the recovery strategy for an error."""
        strategy = rule.recovery_strategy
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                return await self._handle_retry(rule, context)
            elif strategy == RecoveryStrategy.FALLBACK:
                return await self._handle_fallback(rule, context)
            elif strategy == RecoveryStrategy.SKIP:
                return await self._handle_skip(rule, context)
            elif strategy == RecoveryStrategy.RESTART_COMPONENT:
                return await self._handle_restart_component(rule, context)
            elif strategy == RecoveryStrategy.RESTART_PIPELINE:
                return await self._handle_restart_pipeline(rule, context)
            elif strategy == RecoveryStrategy.FAIL:
                return await self._handle_fail(rule, context)
            else:
                self.logger.error(f"Unknown recovery strategy: {strategy}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during recovery strategy execution: {e}")
            return False
    
    async def _handle_retry(self, rule: ErrorRule, context: ErrorContext) -> bool:
        """Handle retry recovery strategy."""
        retry_key = f"{context.stage.value}_{type(context.error).__name__}"
        current_attempts = self.retry_counts.get(retry_key, 0)
        
        retry_config = rule.retry_config or self.default_retry_config
        
        if current_attempts >= retry_config.max_attempts:
            self.logger.error(
                f"Max retry attempts ({retry_config.max_attempts}) exceeded for {retry_key}"
            )
            self.retry_counts[retry_key] = 0  # Reset for future
            return False
        
        # Increment retry count
        self.retry_counts[retry_key] = current_attempts + 1
        
        # Calculate delay
        delay = retry_config.get_delay(current_attempts)
        
        self.logger.info(
            f"Retrying {retry_key} (attempt {current_attempts + 1}/{retry_config.max_attempts}) "
            f"after {delay:.2f}s delay"
        )
        
        # Wait before retry
        await asyncio.sleep(delay)
        
        # If we get here, the retry will be handled by the calling code
        return True
    
    async def _handle_fallback(self, rule: ErrorRule, context: ErrorContext) -> bool:
        """Handle fallback recovery strategy."""
        if rule.fallback_action:
            try:
                self.logger.info(f"Executing fallback action for {context.stage.value}")
                result = await rule.fallback_action(context)
                return result if isinstance(result, bool) else True
            except Exception as e:
                self.logger.error(f"Fallback action failed: {e}")
                return False
        else:
            self.logger.warning(f"No fallback action defined for {context.stage.value}")
            return False
    
    async def _handle_skip(self, rule: ErrorRule, context: ErrorContext) -> bool:
        """Handle skip recovery strategy."""
        self.logger.info(f"Skipping failed operation in {context.stage.value}")
        return True
    
    async def _handle_restart_component(self, rule: ErrorRule, context: ErrorContext) -> bool:
        """Handle component restart recovery strategy."""
        component_key = context.stage.value
        
        # Check restart limits
        if not self._can_restart_component(component_key):
            self.logger.error(f"Component restart limit exceeded for {component_key}")
            return False
        
        self.logger.info(f"Restarting component: {component_key}")
        
        # Record restart attempt
        self.component_restart_counts[component_key] = (
            self.component_restart_counts.get(component_key, 0) + 1
        )
        
        # The actual restart logic would be handled by the pipeline
        # This just signals that a restart should be attempted
        return True
    
    async def _handle_restart_pipeline(self, rule: ErrorRule, context: ErrorContext) -> bool:
        """Handle pipeline restart recovery strategy."""
        self.logger.critical("Pipeline restart requested due to critical error")
        # The actual restart would be handled by the pipeline orchestrator
        return False  # Signal that current operation should stop
    
    async def _handle_fail(self, rule: ErrorRule, context: ErrorContext) -> bool:
        """Handle fail recovery strategy."""
        self.logger.error(f"Failing operation in {context.stage.value} as per error rule")
        return False
    
    def _can_restart_component(self, component_key: str) -> bool:
        """Check if component can be restarted based on limits."""
        restart_count = self.component_restart_counts.get(component_key, 0)
        return restart_count < self.max_component_restarts
    
    async def _fallback_to_cpu(self, context: ErrorContext) -> bool:
        """Fallback action to switch from GPU to CPU processing."""
        self.logger.info(f"Falling back to CPU processing for {context.stage.value}")
        
        # This would typically involve reconfiguring the component to use CPU
        # The actual implementation would depend on the specific component
        context.metadata['fallback_device'] = 'cpu'
        
        return True
    
    async def _fallback_audio_device(self, context: ErrorContext) -> bool:
        """Fallback action to try alternative audio device."""
        self.logger.info(f"Attempting to use alternative audio device for {context.stage.value}")
        
        # This would typically involve trying the next available audio device
        context.metadata['try_alternative_device'] = True
        
        return True
    
    def get_error_statistics(self, duration: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get error statistics for the specified duration."""
        if duration:
            cutoff_time = datetime.now() - duration
            relevant_errors = [e for e in self.error_history if e.timestamp >= cutoff_time]
        else:
            relevant_errors = self.error_history
        
        if not relevant_errors:
            return {
                'total_errors': 0,
                'error_rate': 0.0,
                'errors_by_stage': {},
                'errors_by_severity': {},
                'most_common_errors': []
            }
        
        # Count errors by stage
        errors_by_stage = {}
        for error in relevant_errors:
            stage = error.stage.value
            errors_by_stage[stage] = errors_by_stage.get(stage, 0) + 1
        
        # Count errors by severity
        errors_by_severity = {}
        for error in relevant_errors:
            severity = error.severity.value
            errors_by_severity[severity] = errors_by_severity.get(severity, 0) + 1
        
        # Find most common error types
        error_types = {}
        for error in relevant_errors:
            error_type = type(error.error).__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        most_common_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate error rate (errors per hour)
        if duration:
            hours = duration.total_seconds() / 3600
            error_rate = len(relevant_errors) / max(hours, 1/3600)  # Minimum 1 second
        else:
            error_rate = 0.0
        
        return {
            'total_errors': len(relevant_errors),
            'error_rate': error_rate,
            'errors_by_stage': errors_by_stage,
            'errors_by_severity': errors_by_severity,
            'most_common_errors': most_common_errors
        }
    
    def clear_error_history(self):
        """Clear the error history."""
        self.error_history.clear()
        self.retry_counts.clear()
        self.component_restart_counts.clear()
        self.logger.info("Error history cleared")
    
    def get_recent_errors(self, count: int = 10) -> List[ErrorContext]:
        """Get the most recent errors."""
        return self.error_history[-count:] if self.error_history else []


class GracefulDegradationManager:
    """Manages graceful degradation of system capabilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.degradation_state: Dict[str, Any] = {}
        self.fallback_configs: Dict[str, Dict[str, Any]] = {}
        
        # Setup default fallback configurations
        self._setup_default_fallbacks()
    
    def _setup_default_fallbacks(self):
        """Setup default fallback configurations."""
        
        # GPU to CPU fallback
        self.fallback_configs['gpu_to_cpu'] = {
            'stt_model_size': 'tiny',  # Use smaller model on CPU
            'llm_max_tokens': 128,     # Reduce token limit
            'tts_quality': 'low',      # Use lower quality TTS
            'batch_size': 1            # Process one item at a time
        }
        
        # High quality to low quality fallback
        self.fallback_configs['quality_degradation'] = {
            'stt_model_size': 'base',  # Use base instead of large
            'audio_sample_rate': 16000, # Lower sample rate
            'tts_quality': 'medium',   # Medium quality TTS
            'character_intensity': 0.5  # Reduce character transformation intensity
        }
        
        # Performance degradation fallback
        self.fallback_configs['performance_degradation'] = {
            'max_queue_size': 5,       # Smaller queues
            'processing_timeout': 30,  # Longer timeouts
            'enable_caching': True,    # Enable aggressive caching
            'skip_non_essential': True # Skip non-essential processing
        }
    
    def enable_degradation(self, degradation_type: str, reason: str):
        """Enable a specific type of degradation."""
        if degradation_type in self.fallback_configs:
            self.degradation_state[degradation_type] = {
                'enabled': True,
                'reason': reason,
                'timestamp': datetime.now(),
                'config': self.fallback_configs[degradation_type]
            }
            
            self.logger.warning(
                f"Enabled degradation '{degradation_type}' due to: {reason}"
            )
        else:
            self.logger.error(f"Unknown degradation type: {degradation_type}")
    
    def disable_degradation(self, degradation_type: str):
        """Disable a specific type of degradation."""
        if degradation_type in self.degradation_state:
            del self.degradation_state[degradation_type]
            self.logger.info(f"Disabled degradation '{degradation_type}'")
    
    def is_degraded(self, degradation_type: str) -> bool:
        """Check if a specific degradation is active."""
        return degradation_type in self.degradation_state
    
    def get_degraded_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get configuration with degradation applied."""
        config = base_config.copy()
        
        for degradation_type, degradation_info in self.degradation_state.items():
            if degradation_info['enabled']:
                fallback_config = degradation_info['config']
                config.update(fallback_config)
        
        return config
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status."""
        return {
            'active_degradations': list(self.degradation_state.keys()),
            'degradation_details': self.degradation_state,
            'available_degradations': list(self.fallback_configs.keys())
        }