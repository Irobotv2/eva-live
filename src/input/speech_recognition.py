"""
Eva Live Speech Recognition Module

This module handles speech-to-text conversion using multiple providers
with fallback mechanisms for reliability. Supports Google Cloud Speech-to-Text
as primary and OpenAI Whisper as fallback.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Callable, AsyncGenerator
from dataclasses import dataclass
import numpy as np

# Audio processing
import pyaudio
import wave
import io

# AI services
import openai
from google.cloud import speech
import whisper

from ..shared.config import get_config
from ..shared.models import PerformanceMetric

@dataclass
class AudioConfig:
    """Audio configuration settings"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: int = pyaudio.paInt16
    encoding: str = "LINEAR16"

@dataclass
class TranscriptionResult:
    """Speech recognition result"""
    text: str
    confidence: float
    language: str
    processing_time_ms: int
    provider: str
    alternatives: List[str] = None
    is_final: bool = True

class VoiceActivityDetector:
    """Simple voice activity detection using energy thresholds"""
    
    def __init__(self, threshold: float = 0.01, frame_duration_ms: int = 30):
        self.threshold = threshold
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(16000 * frame_duration_ms / 1000)  # 16kHz sample rate
    
    def is_speech(self, audio_data: bytes) -> bool:
        """Determine if audio contains speech using energy detection"""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate RMS energy
            if len(audio_array) == 0:
                return False
            
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            
            # Normalize by maximum possible value
            normalized_rms = rms / 32768.0
            
            return normalized_rms > self.threshold
        except Exception as e:
            logging.warning(f"VAD error: {e}")
            return True  # Default to assuming speech

class SpeechRecognitionProvider(ABC):
    """Abstract base class for speech recognition providers"""
    
    @abstractmethod
    async def transcribe_stream(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[TranscriptionResult, None]:
        """Transcribe streaming audio"""
        pass
    
    @abstractmethod
    async def transcribe_audio(self, audio_data: bytes, config: AudioConfig) -> TranscriptionResult:
        """Transcribe audio data"""
        pass

class GoogleSpeechProvider(SpeechRecognitionProvider):
    """Google Cloud Speech-to-Text provider"""
    
    def __init__(self):
        self.config = get_config()
        self.client = speech.SpeechClient()
        self.logger = logging.getLogger(__name__)
    
    async def transcribe_stream(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[TranscriptionResult, None]:
        """Transcribe streaming audio using Google Cloud Speech"""
        try:
            # Configure streaming recognition
            recognition_config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=self.config.get('ai_services.speech_recognition.primary.language_code', 'en-US'),
                enable_automatic_punctuation=True,
                enable_word_confidence=True,
                model="latest_long"
            )
            
            streaming_config = speech.StreamingRecognitionConfig(
                config=recognition_config,
                interim_results=True,
                single_utterance=False
            )
            
            # Create audio generator
            async def audio_generator():
                yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
                async for chunk in audio_stream:
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)
            
            # Process streaming responses
            responses = self.client.streaming_recognize(audio_generator())
            
            for response in responses:
                for result in response.results:
                    if result.alternatives:
                        alternative = result.alternatives[0]
                        
                        yield TranscriptionResult(
                            text=alternative.transcript,
                            confidence=alternative.confidence if hasattr(alternative, 'confidence') else 0.0,
                            language='en-US',
                            processing_time_ms=0,  # Real-time streaming
                            provider='google',
                            is_final=result.is_final
                        )
        
        except Exception as e:
            self.logger.error(f"Google Speech streaming error: {e}")
            raise
    
    async def transcribe_audio(self, audio_data: bytes, config: AudioConfig) -> TranscriptionResult:
        """Transcribe audio data using Google Cloud Speech"""
        start_time = time.time()
        
        try:
            # Configure recognition
            recognition_config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=config.sample_rate,
                language_code=self.config.get('ai_services.speech_recognition.primary.language_code', 'en-US'),
                enable_automatic_punctuation=True,
                enable_word_confidence=True,
                alternative_language_codes=['en-GB', 'es-US']  # Additional language support
            )
            
            audio = speech.RecognitionAudio(content=audio_data)
            
            # Perform recognition
            response = self.client.recognize(config=recognition_config, audio=audio)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            if response.results:
                result = response.results[0]
                alternative = result.alternatives[0]
                
                # Extract alternatives
                alternatives = [alt.transcript for alt in result.alternatives[1:5]]
                
                return TranscriptionResult(
                    text=alternative.transcript,
                    confidence=alternative.confidence,
                    language='en-US',
                    processing_time_ms=processing_time,
                    provider='google',
                    alternatives=alternatives
                )
            else:
                return TranscriptionResult(
                    text="",
                    confidence=0.0,
                    language='en-US',
                    processing_time_ms=processing_time,
                    provider='google'
                )
        
        except Exception as e:
            self.logger.error(f"Google Speech recognition error: {e}")
            raise

class WhisperProvider(SpeechRecognitionProvider):
    """OpenAI Whisper provider (local or API)"""
    
    def __init__(self, use_api: bool = True):
        self.config = get_config()
        self.use_api = use_api
        self.logger = logging.getLogger(__name__)
        
        if not use_api:
            # Load local Whisper model
            model_name = self.config.get('ai_services.speech_recognition.fallback.model', 'base')
            self.model = whisper.load_model(model_name)
        else:
            openai.api_key = self.config.ai_services.openai_api_key
    
    async def transcribe_stream(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[TranscriptionResult, None]:
        """Transcribe streaming audio using Whisper (buffered approach)"""
        buffer = io.BytesIO()
        buffer_duration = 5.0  # Process every 5 seconds
        sample_rate = 16000
        buffer_size = int(sample_rate * buffer_duration * 2)  # 2 bytes per sample
        
        try:
            async for chunk in audio_stream:
                buffer.write(chunk)
                
                if buffer.tell() >= buffer_size:
                    # Process buffer
                    audio_data = buffer.getvalue()
                    buffer = io.BytesIO()  # Reset buffer
                    
                    result = await self.transcribe_audio(audio_data, AudioConfig())
                    if result.text.strip():
                        yield result
        
        except Exception as e:
            self.logger.error(f"Whisper streaming error: {e}")
            raise
    
    async def transcribe_audio(self, audio_data: bytes, config: AudioConfig) -> TranscriptionResult:
        """Transcribe audio data using Whisper"""
        start_time = time.time()
        
        try:
            if self.use_api:
                # Use OpenAI Whisper API
                audio_file = io.BytesIO(audio_data)
                audio_file.name = "audio.wav"
                
                response = await openai.Audio.atranscribe(
                    model="whisper-1",
                    file=audio_file,
                    language=self.config.get('ai_services.speech_recognition.fallback.language', 'en')
                )
                
                text = response['text']
                confidence = 0.85  # Whisper doesn't provide confidence scores
                
            else:
                # Use local Whisper model
                # Convert bytes to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0
                
                # Transcribe
                result = self.model.transcribe(audio_float)
                text = result['text']
                confidence = 0.8  # Estimate confidence
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return TranscriptionResult(
                text=text,
                confidence=confidence,
                language='en',
                processing_time_ms=processing_time,
                provider='whisper'
            )
        
        except Exception as e:
            self.logger.error(f"Whisper recognition error: {e}")
            raise

class SpeechRecognitionModule:
    """Main speech recognition module with fallback support"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.config = get_config()
        self.session_id = session_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize providers
        self.primary_provider = GoogleSpeechProvider()
        self.fallback_provider = WhisperProvider(use_api=True)
        
        # Initialize VAD
        self.vad = VoiceActivityDetector()
        
        # Performance tracking
        self.metrics: List[PerformanceMetric] = []
        
        # Callbacks
        self.on_transcription: Optional[Callable[[TranscriptionResult], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
    
    async def transcribe_stream(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[TranscriptionResult, None]:
        """Transcribe streaming audio with fallback"""
        try:
            # Try primary provider first
            async for result in self.primary_provider.transcribe_stream(audio_stream):
                # Record metrics
                await self._record_metric("latency_ms", result.processing_time_ms, "speech_recognition")
                await self._record_metric("confidence_score", result.confidence, "speech_recognition")
                
                yield result
                
                # Call callback if set
                if self.on_transcription:
                    self.on_transcription(result)
        
        except Exception as e:
            self.logger.warning(f"Primary provider failed, using fallback: {e}")
            
            try:
                # Use fallback provider
                async for result in self.fallback_provider.transcribe_stream(audio_stream):
                    await self._record_metric("latency_ms", result.processing_time_ms, "speech_recognition_fallback")
                    await self._record_metric("confidence_score", result.confidence, "speech_recognition_fallback")
                    
                    yield result
                    
                    if self.on_transcription:
                        self.on_transcription(result)
            
            except Exception as fallback_error:
                self.logger.error(f"Both providers failed: {fallback_error}")
                if self.on_error:
                    self.on_error(fallback_error)
                raise
    
    async def transcribe_audio(self, audio_data: bytes) -> TranscriptionResult:
        """Transcribe audio data with fallback"""
        # Check if audio contains speech
        if not self.vad.is_speech(audio_data):
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language="en-US",
                processing_time_ms=0,
                provider="vad"
            )
        
        config = AudioConfig()
        
        try:
            # Try primary provider
            result = await self.primary_provider.transcribe_audio(audio_data, config)
            
            # Record metrics
            await self._record_metric("latency_ms", result.processing_time_ms, "speech_recognition")
            await self._record_metric("confidence_score", result.confidence, "speech_recognition")
            
            return result
        
        except Exception as e:
            self.logger.warning(f"Primary provider failed, using fallback: {e}")
            
            try:
                # Use fallback provider
                result = await self.fallback_provider.transcribe_audio(audio_data, config)
                
                await self._record_metric("latency_ms", result.processing_time_ms, "speech_recognition_fallback")
                await self._record_metric("confidence_score", result.confidence, "speech_recognition_fallback")
                
                return result
            
            except Exception as fallback_error:
                self.logger.error(f"Both providers failed: {fallback_error}")
                raise
    
    async def start_real_time_recognition(self, audio_callback: Callable[[TranscriptionResult], None]) -> None:
        """Start real-time audio recognition from microphone"""
        self.on_transcription = audio_callback
        
        # Audio configuration
        config = AudioConfig()
        
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        
        try:
            # Open audio stream
            stream = p.open(
                format=config.format,
                channels=config.channels,
                rate=config.sample_rate,
                input=True,
                frames_per_buffer=config.chunk_size
            )
            
            self.logger.info("Started real-time speech recognition")
            
            # Create audio generator
            async def audio_generator():
                while True:
                    try:
                        data = stream.read(config.chunk_size, exception_on_overflow=False)
                        yield data
                        await asyncio.sleep(0.01)  # Small delay to prevent blocking
                    except Exception as e:
                        self.logger.error(f"Audio stream error: {e}")
                        break
            
            # Process audio stream
            async for result in self.transcribe_stream(audio_generator()):
                if result.text.strip():  # Only process non-empty results
                    self.logger.info(f"Transcription: {result.text} (confidence: {result.confidence:.2f})")
        
        except Exception as e:
            self.logger.error(f"Real-time recognition error: {e}")
            raise
        
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
            p.terminate()
    
    async def _record_metric(self, metric_type: str, value: float, component: str) -> None:
        """Record performance metric"""
        metric = PerformanceMetric(
            metric_type=metric_type,
            metric_value=value,
            component=component
        )
        
        if self.session_id:
            # In a real implementation, this would be saved to database
            pass
        
        self.metrics.append(metric)
    
    def get_metrics(self) -> List[PerformanceMetric]:
        """Get recorded metrics"""
        return self.metrics.copy()

# Utility functions
def create_wav_header(sample_rate: int, channels: int, bits_per_sample: int, data_size: int) -> bytes:
    """Create WAV file header"""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    
    header = b'RIFF'
    header += (36 + data_size).to_bytes(4, 'little')
    header += b'WAVE'
    header += b'fmt '
    header += (16).to_bytes(4, 'little')
    header += (1).to_bytes(2, 'little')  # PCM format
    header += channels.to_bytes(2, 'little')
    header += sample_rate.to_bytes(4, 'little')
    header += byte_rate.to_bytes(4, 'little')
    header += block_align.to_bytes(2, 'little')
    header += bits_per_sample.to_bytes(2, 'little')
    header += b'data'
    header += data_size.to_bytes(4, 'little')
    
    return header

async def test_speech_recognition():
    """Test function for speech recognition module"""
    import asyncio
    
    # Initialize module
    module = SpeechRecognitionModule()
    
    # Test with sample audio (you would need to provide actual audio data)
    test_audio = b'\x00' * 16000 * 2  # 1 second of silence
    
    try:
        result = await module.transcribe_audio(test_audio)
        print(f"Transcription: {result.text}")
        print(f"Confidence: {result.confidence}")
        print(f"Processing time: {result.processing_time_ms}ms")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_speech_recognition())
