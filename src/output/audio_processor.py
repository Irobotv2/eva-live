"""
Eva Live Audio Processing Module

This module handles audio format conversion, optimization, noise reduction,
and real-time audio streaming for the Eva Live system.
"""

import asyncio
import logging
import time
import io
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Audio processing libraries
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import soundfile as sf
import librosa
import noisereduce as nr

from ..shared.config import get_config
from ..shared.models import PerformanceMetric

class AudioFormat(str, Enum):
    """Supported audio formats"""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    FLAC = "flac"
    AAC = "aac"
    M4A = "m4a"

class AudioQuality(str, Enum):
    """Audio quality presets"""
    LOW = "low"          # 22kHz, 64kbps
    MEDIUM = "medium"    # 44kHz, 128kbps  
    HIGH = "high"        # 48kHz, 192kbps
    ULTRA = "ultra"      # 48kHz, 320kbps

class ProcessingEffect(str, Enum):
    """Available audio processing effects"""
    NORMALIZE = "normalize"
    COMPRESS = "compress"
    NOISE_REDUCE = "noise_reduce"
    ENHANCE = "enhance"
    EQ = "eq"
    REVERB = "reverb"

@dataclass
class AudioSettings:
    """Audio processing settings"""
    sample_rate: int = 44100
    channels: int = 1  # Mono by default
    bit_depth: int = 16
    quality: AudioQuality = AudioQuality.HIGH
    format: AudioFormat = AudioFormat.MP3
    effects: List[ProcessingEffect] = None
    
@dataclass 
class AudioMetrics:
    """Audio quality and processing metrics"""
    duration_ms: int
    sample_rate: int
    channels: int
    bit_depth: int
    file_size_bytes: int
    peak_amplitude: float
    rms_level: float
    dynamic_range: float
    noise_floor: float
    processing_time_ms: int

@dataclass
class ProcessedAudio:
    """Processed audio result"""
    audio_data: bytes
    format: AudioFormat
    settings: AudioSettings
    metrics: AudioMetrics
    success: bool
    error_message: Optional[str] = None

class AudioProcessor:
    """Main audio processing engine"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.config = get_config()
        self.session_id = session_id
        self.logger = logging.getLogger(__name__)
        
        # Quality presets
        self.quality_presets = {
            AudioQuality.LOW: {
                'sample_rate': 22050,
                'bitrate': '64k',
                'channels': 1
            },
            AudioQuality.MEDIUM: {
                'sample_rate': 44100,
                'bitrate': '128k',
                'channels': 1
            },
            AudioQuality.HIGH: {
                'sample_rate': 48000,
                'bitrate': '192k',
                'channels': 1
            },
            AudioQuality.ULTRA: {
                'sample_rate': 48000,
                'bitrate': '320k',
                'channels': 2
            }
        }
        
        # Performance tracking
        self.metrics: List[PerformanceMetric] = []
    
    async def process_audio(
        self,
        audio_data: bytes,
        input_format: AudioFormat,
        settings: Optional[AudioSettings] = None
    ) -> ProcessedAudio:
        """Process audio with specified settings"""
        start_time = time.time()
        
        try:
            # Use default settings if none provided
            if settings is None:
                settings = AudioSettings()
            
            # Load audio
            audio_segment = self._load_audio(audio_data, input_format)
            
            # Apply quality preset
            preset = self.quality_presets[settings.quality]
            target_sample_rate = preset['sample_rate']
            target_channels = preset['channels']
            
            # Resample if needed
            if audio_segment.frame_rate != target_sample_rate:
                audio_segment = audio_segment.set_frame_rate(target_sample_rate)
            
            # Set channels
            if settings.channels != audio_segment.channels:
                if settings.channels == 1:
                    audio_segment = audio_segment.set_channels(1)
                elif settings.channels == 2:
                    audio_segment = audio_segment.set_channels(2)
            
            # Apply effects
            if settings.effects:
                audio_segment = await self._apply_effects(audio_segment, settings.effects)
            
            # Convert to target format
            output_data = self._export_audio(audio_segment, settings.format, preset)
            
            # Calculate metrics
            metrics = self._calculate_metrics(audio_segment, output_data, start_time)
            
            # Record performance metrics
            await self._record_metric("audio_processing_time_ms", metrics.processing_time_ms, "audio_processor")
            await self._record_metric("audio_duration_ms", metrics.duration_ms, "audio_processor")
            await self._record_metric("audio_quality_score", self._calculate_quality_score(metrics), "audio_processor")
            
            self.logger.info(f"Audio processed: {metrics.duration_ms}ms duration in {metrics.processing_time_ms}ms")
            
            return ProcessedAudio(
                audio_data=output_data,
                format=settings.format,
                settings=settings,
                metrics=metrics,
                success=True
            )
            
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"Audio processing failed: {e}")
            
            return ProcessedAudio(
                audio_data=b"",
                format=settings.format if settings else AudioFormat.MP3,
                settings=settings or AudioSettings(),
                metrics=AudioMetrics(
                    duration_ms=0,
                    sample_rate=0,
                    channels=0,
                    bit_depth=0,
                    file_size_bytes=0,
                    peak_amplitude=0.0,
                    rms_level=0.0,
                    dynamic_range=0.0,
                    noise_floor=0.0,
                    processing_time_ms=processing_time
                ),
                success=False,
                error_message=str(e)
            )
    
    async def convert_format(
        self,
        audio_data: bytes,
        input_format: AudioFormat,
        output_format: AudioFormat,
        quality: AudioQuality = AudioQuality.HIGH
    ) -> ProcessedAudio:
        """Convert audio between formats"""
        settings = AudioSettings(
            format=output_format,
            quality=quality
        )
        
        return await self.process_audio(audio_data, input_format, settings)
    
    async def optimize_for_streaming(
        self,
        audio_data: bytes,
        input_format: AudioFormat
    ) -> ProcessedAudio:
        """Optimize audio for real-time streaming"""
        settings = AudioSettings(
            sample_rate=22050,  # Lower sample rate for streaming
            channels=1,         # Mono for bandwidth efficiency
            quality=AudioQuality.MEDIUM,
            format=AudioFormat.MP3,
            effects=[ProcessingEffect.NORMALIZE, ProcessingEffect.COMPRESS]
        )
        
        return await self.process_audio(audio_data, input_format, settings)
    
    async def enhance_speech(
        self,
        audio_data: bytes,
        input_format: AudioFormat
    ) -> ProcessedAudio:
        """Enhance speech audio for clarity"""
        settings = AudioSettings(
            sample_rate=16000,  # Optimal for speech
            channels=1,
            quality=AudioQuality.HIGH,
            format=AudioFormat.WAV,
            effects=[
                ProcessingEffect.NOISE_REDUCE,
                ProcessingEffect.NORMALIZE,
                ProcessingEffect.ENHANCE
            ]
        )
        
        return await self.process_audio(audio_data, input_format, settings)
    
    def _load_audio(self, audio_data: bytes, format: AudioFormat) -> AudioSegment:
        """Load audio data into AudioSegment"""
        try:
            audio_io = io.BytesIO(audio_data)
            
            if format == AudioFormat.MP3:
                return AudioSegment.from_mp3(audio_io)
            elif format == AudioFormat.WAV:
                return AudioSegment.from_wav(audio_io)
            elif format == AudioFormat.OGG:
                return AudioSegment.from_ogg(audio_io)
            elif format == AudioFormat.FLAC:
                return AudioSegment.from_file(audio_io, format="flac")
            else:
                # Try generic loader
                return AudioSegment.from_file(audio_io, format=format.value)
                
        except Exception as e:
            self.logger.error(f"Failed to load audio format {format}: {e}")
            raise
    
    def _export_audio(self, audio_segment: AudioSegment, format: AudioFormat, preset: Dict[str, Any]) -> bytes:
        """Export audio to specified format"""
        try:
            output_io = io.BytesIO()
            
            # Export parameters
            export_params = {
                'format': format.value,
                'parameters': []
            }
            
            # Add bitrate for compressed formats
            if format in [AudioFormat.MP3, AudioFormat.OGG, AudioFormat.AAC]:
                export_params['parameters'].extend(['-b:a', preset['bitrate']])
            
            # Export audio
            audio_segment.export(
                output_io,
                **export_params
            )
            
            return output_io.getvalue()
            
        except Exception as e:
            self.logger.error(f"Failed to export audio format {format}: {e}")
            raise
    
    async def _apply_effects(self, audio_segment: AudioSegment, effects: List[ProcessingEffect]) -> AudioSegment:
        """Apply audio effects"""
        processed = audio_segment
        
        for effect in effects:
            try:
                if effect == ProcessingEffect.NORMALIZE:
                    processed = normalize(processed)
                    
                elif effect == ProcessingEffect.COMPRESS:
                    processed = compress_dynamic_range(processed, threshold=-20.0, ratio=4.0)
                    
                elif effect == ProcessingEffect.NOISE_REDUCE:
                    processed = await self._apply_noise_reduction(processed)
                    
                elif effect == ProcessingEffect.ENHANCE:
                    processed = await self._apply_speech_enhancement(processed)
                    
                elif effect == ProcessingEffect.EQ:
                    processed = await self._apply_eq(processed)
                    
                self.logger.debug(f"Applied effect: {effect}")
                
            except Exception as e:
                self.logger.warning(f"Failed to apply effect {effect}: {e}")
                # Continue with other effects
        
        return processed
    
    async def _apply_noise_reduction(self, audio_segment: AudioSegment) -> AudioSegment:
        """Apply noise reduction using spectral subtraction"""
        try:
            # Convert to numpy array
            samples = np.array(audio_segment.get_array_of_samples())
            
            # Apply noise reduction
            reduced = nr.reduce_noise(y=samples, sr=audio_segment.frame_rate)
            
            # Convert back to AudioSegment
            reduced_segment = AudioSegment(
                reduced.tobytes(),
                frame_rate=audio_segment.frame_rate,
                sample_width=audio_segment.sample_width,
                channels=audio_segment.channels
            )
            
            return reduced_segment
            
        except Exception as e:
            self.logger.warning(f"Noise reduction failed: {e}")
            return audio_segment
    
    async def _apply_speech_enhancement(self, audio_segment: AudioSegment) -> AudioSegment:
        """Apply speech-specific enhancement"""
        try:
            # Convert to numpy for processing
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            samples = samples / np.max(np.abs(samples))  # Normalize
            
            # Apply spectral enhancement for speech
            enhanced = librosa.effects.preemphasis(samples)
            
            # Convert back
            enhanced = (enhanced * 32767).astype(np.int16)
            enhanced_segment = AudioSegment(
                enhanced.tobytes(),
                frame_rate=audio_segment.frame_rate,
                sample_width=2,
                channels=audio_segment.channels
            )
            
            return enhanced_segment
            
        except Exception as e:
            self.logger.warning(f"Speech enhancement failed: {e}")
            return audio_segment
    
    async def _apply_eq(self, audio_segment: AudioSegment) -> AudioSegment:
        """Apply equalization for speech clarity"""
        try:
            # Simple high-pass filter to remove low-frequency noise
            # This is a basic implementation - could be enhanced with more sophisticated EQ
            
            # Apply high-pass filter effect (boost frequencies above 80Hz)
            filtered = audio_segment.high_pass_filter(80)
            
            # Slight boost in speech frequencies (1-4kHz)
            # This is approximated by adjusting gain
            boosted = filtered + 1  # +1dB boost
            
            return boosted
            
        except Exception as e:
            self.logger.warning(f"EQ application failed: {e}")
            return audio_segment
    
    def _calculate_metrics(self, audio_segment: AudioSegment, output_data: bytes, start_time: float) -> AudioMetrics:
        """Calculate audio quality metrics"""
        try:
            # Basic metrics
            duration_ms = len(audio_segment)
            sample_rate = audio_segment.frame_rate
            channels = audio_segment.channels
            bit_depth = audio_segment.sample_width * 8
            file_size_bytes = len(output_data)
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Convert to numpy for analysis
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            
            # Normalize samples
            if len(samples) > 0:
                samples = samples / (2 ** (bit_depth - 1))
                
                # Calculate metrics
                peak_amplitude = float(np.max(np.abs(samples)))
                rms_level = float(np.sqrt(np.mean(samples ** 2)))
                
                # Dynamic range (difference between peak and RMS in dB)
                if rms_level > 0:
                    dynamic_range = 20 * np.log10(peak_amplitude / rms_level)
                else:
                    dynamic_range = 0.0
                
                # Estimate noise floor (bottom 10% of amplitude distribution)
                sorted_samples = np.sort(np.abs(samples))
                noise_floor_idx = int(len(sorted_samples) * 0.1)
                noise_floor = float(np.mean(sorted_samples[:noise_floor_idx]))
                
            else:
                peak_amplitude = 0.0
                rms_level = 0.0
                dynamic_range = 0.0
                noise_floor = 0.0
            
            return AudioMetrics(
                duration_ms=duration_ms,
                sample_rate=sample_rate,
                channels=channels,
                bit_depth=bit_depth,
                file_size_bytes=file_size_bytes,
                peak_amplitude=peak_amplitude,
                rms_level=rms_level,
                dynamic_range=dynamic_range,
                noise_floor=noise_floor,
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate audio metrics: {e}")
            return AudioMetrics(
                duration_ms=0,
                sample_rate=0,
                channels=0,
                bit_depth=0,
                file_size_bytes=len(output_data),
                peak_amplitude=0.0,
                rms_level=0.0,
                dynamic_range=0.0,
                noise_floor=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
    
    def _calculate_quality_score(self, metrics: AudioMetrics) -> float:
        """Calculate overall audio quality score (0-1)"""
        try:
            score = 0.0
            
            # Sample rate score (higher is better, up to 48kHz)
            sr_score = min(metrics.sample_rate / 48000, 1.0) * 0.25
            
            # Dynamic range score (good speech should have 20-40dB dynamic range)
            dr_target = 30.0
            dr_score = max(0, 1.0 - abs(metrics.dynamic_range - dr_target) / dr_target) * 0.25
            
            # RMS level score (good speech around 0.1-0.3 RMS)
            rms_target = 0.2
            rms_score = max(0, 1.0 - abs(metrics.rms_level - rms_target) / rms_target) * 0.25
            
            # Noise floor score (lower is better)
            noise_score = max(0, 1.0 - metrics.noise_floor) * 0.25
            
            score = sr_score + dr_score + rms_score + noise_score
            
            return min(max(score, 0.0), 1.0)  # Clamp to 0-1
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate quality score: {e}")
            return 0.5  # Default neutral score
    
    async def analyze_audio_quality(self, audio_data: bytes, format: AudioFormat) -> Dict[str, Any]:
        """Analyze audio quality and provide recommendations"""
        try:
            # Load and analyze audio
            audio_segment = self._load_audio(audio_data, format)
            
            # Calculate metrics
            metrics = self._calculate_metrics(audio_segment, audio_data, time.time())
            quality_score = self._calculate_quality_score(metrics)
            
            # Generate recommendations
            recommendations = []
            
            if metrics.sample_rate < 16000:
                recommendations.append("Consider increasing sample rate to at least 16kHz for better speech quality")
            
            if metrics.rms_level < 0.05:
                recommendations.append("Audio level is too low, consider amplification")
            elif metrics.rms_level > 0.5:
                recommendations.append("Audio level is too high, consider attenuation")
            
            if metrics.dynamic_range < 10:
                recommendations.append("Low dynamic range detected, audio may sound compressed")
            
            if metrics.noise_floor > 0.1:
                recommendations.append("High noise floor detected, consider noise reduction")
            
            return {
                'quality_score': quality_score,
                'metrics': metrics.__dict__,
                'recommendations': recommendations,
                'overall_assessment': self._get_quality_assessment(quality_score)
            }
            
        except Exception as e:
            self.logger.error(f"Audio quality analysis failed: {e}")
            return {
                'quality_score': 0.0,
                'metrics': {},
                'recommendations': ["Audio analysis failed"],
                'overall_assessment': "Unable to assess"
            }
    
    def _get_quality_assessment(self, score: float) -> str:
        """Get quality assessment based on score"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        elif score >= 0.2:
            return "Poor"
        else:
            return "Very Poor"
    
    async def _record_metric(self, metric_type: str, value: float, component: str) -> None:
        """Record performance metric"""
        metric = PerformanceMetric(
            metric_type=metric_type,
            metric_value=value,
            component=component
        )
        self.metrics.append(metric)
    
    def get_metrics(self) -> List[PerformanceMetric]:
        """Get recorded metrics"""
        return self.metrics.copy()

# Utility functions
async def process_audio_async(
    audio_data: bytes,
    input_format: AudioFormat,
    settings: Optional[AudioSettings] = None
) -> ProcessedAudio:
    """Convenience function to process audio"""
    processor = AudioProcessor()
    return await processor.process_audio(audio_data, input_format, settings)

async def test_audio_processor():
    """Test function for audio processor"""
    try:
        # Create a simple test tone
        from pydub.generators import Sine
        
        # Generate 1 second 440Hz tone
        test_tone = Sine(440).to_audio_segment(duration=1000)
        
        # Export to bytes
        test_data = io.BytesIO()
        test_tone.export(test_data, format="wav")
        test_audio_data = test_data.getvalue()
        
        # Test processing
        processor = AudioProcessor()
        
        # Test basic processing
        result = await processor.process_audio(
            test_audio_data, 
            AudioFormat.WAV,
            AudioSettings(quality=AudioQuality.HIGH, format=AudioFormat.MP3)
        )
        
        print(f"Processing successful: {result.success}")
        print(f"Duration: {result.metrics.duration_ms}ms")
        print(f"Sample rate: {result.metrics.sample_rate}Hz")
        print(f"Processing time: {result.metrics.processing_time_ms}ms")
        print(f"Quality score: {processor._calculate_quality_score(result.metrics):.2f}")
        
        # Test format conversion
        converted = await processor.convert_format(
            test_audio_data,
            AudioFormat.WAV,
            AudioFormat.MP3
        )
        
        print(f"Format conversion successful: {converted.success}")
        print(f"Output size: {len(converted.audio_data)} bytes")
        
        # Test quality analysis
        analysis = await processor.analyze_audio_quality(test_audio_data, AudioFormat.WAV)
        print(f"Quality analysis: {analysis['overall_assessment']} ({analysis['quality_score']:.2f})")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_audio_processor())
