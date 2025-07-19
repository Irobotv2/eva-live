"""
Eva Live Voice Synthesis Module

This module handles text-to-speech conversion using multiple providers
with fallback mechanisms. Supports ElevenLabs as primary and Azure 
Cognitive Services as fallback.
"""

import asyncio
import logging
import time
import io
import base64
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

# Audio processing
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play

# AI services
import aiohttp
import azure.cognitiveservices.speech as speechsdk

from ..shared.config import get_config
from ..shared.models import PerformanceMetric

class VoiceProvider(str, Enum):
    """Available voice synthesis providers"""
    ELEVENLABS = "elevenlabs"
    AZURE = "azure"
    GOOGLE = "google"

class AudioFormat(str, Enum):
    """Supported audio formats"""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    FLAC = "flac"

class EmotionType(str, Enum):
    """Emotion types for voice modulation"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    EXCITED = "excited"
    CONFIDENT = "confident"
    EMPATHETIC = "empathetic"
    PROFESSIONAL = "professional"

@dataclass
class VoiceSettings:
    """Voice configuration settings"""
    voice_id: str
    stability: float = 0.75  # 0.0 - 1.0
    similarity_boost: float = 0.75  # 0.0 - 1.0
    style: float = 0.0  # 0.0 - 1.0
    speaking_rate: float = 1.0  # 0.25 - 4.0
    pitch: float = 0.0  # -20.0 - 20.0
    volume: float = 1.0  # 0.0 - 2.0
    emotion: EmotionType = EmotionType.NEUTRAL
    
@dataclass
class SynthesisResult:
    """Result of voice synthesis"""
    audio_data: bytes
    format: AudioFormat
    duration_ms: int
    sample_rate: int
    processing_time_ms: int
    provider: VoiceProvider
    voice_settings: VoiceSettings
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None

class VoiceSynthesisProvider(ABC):
    """Abstract base class for voice synthesis providers"""
    
    @abstractmethod
    async def synthesize(self, text: str, voice_settings: VoiceSettings) -> SynthesisResult:
        """Synthesize speech from text"""
        pass
    
    @abstractmethod
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy"""
        pass

class ElevenLabsProvider(VoiceSynthesisProvider):
    """ElevenLabs voice synthesis provider"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        self.api_key = self.config.ai_services.elevenlabs_api_key
        self.base_url = "https://api.elevenlabs.io/v1"
        
        # Voice model settings
        self.model_id = self.config.ai_services.elevenlabs_model_id
        
    async def synthesize(self, text: str, voice_settings: VoiceSettings) -> SynthesisResult:
        """Synthesize speech using ElevenLabs API"""
        start_time = time.time()
        
        try:
            # Prepare request
            url = f"{self.base_url}/text-to-speech/{voice_settings.voice_id}/stream"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            # Map emotions to ElevenLabs style settings
            emotion_settings = self._map_emotion_to_settings(voice_settings.emotion)
            
            data = {
                "text": text,
                "model_id": self.model_id,
                "voice_settings": {
                    "stability": voice_settings.stability,
                    "similarity_boost": voice_settings.similarity_boost,
                    "style": emotion_settings.get("style", voice_settings.style),
                    "use_speaker_boost": True
                }
            }
            
            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, headers=headers, timeout=30) as response:
                    if response.status == 200:
                        audio_data = await response.read()
                        
                        processing_time = int((time.time() - start_time) * 1000)
                        
                        # Get audio duration
                        duration_ms = self._get_audio_duration(audio_data, AudioFormat.MP3)
                        
                        return SynthesisResult(
                            audio_data=audio_data,
                            format=AudioFormat.MP3,
                            duration_ms=duration_ms,
                            sample_rate=22050,  # ElevenLabs default
                            processing_time_ms=processing_time,
                            provider=VoiceProvider.ELEVENLABS,
                            voice_settings=voice_settings,
                            metadata={
                                "model_id": self.model_id,
                                "text_length": len(text),
                                "emotion_mapped": emotion_settings
                            },
                            success=True
                        )
                    else:
                        error_msg = f"ElevenLabs API error: {response.status}"
                        self.logger.error(f"{error_msg} - {await response.text()}")
                        raise Exception(error_msg)
                        
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"ElevenLabs synthesis failed: {e}")
            
            return SynthesisResult(
                audio_data=b"",
                format=AudioFormat.MP3,
                duration_ms=0,
                sample_rate=22050,
                processing_time_ms=processing_time,
                provider=VoiceProvider.ELEVENLABS,
                voice_settings=voice_settings,
                metadata={"error": str(e)},
                success=False,
                error_message=str(e)
            )
    
    def _map_emotion_to_settings(self, emotion: EmotionType) -> Dict[str, float]:
        """Map emotion types to ElevenLabs voice settings"""
        emotion_mappings = {
            EmotionType.NEUTRAL: {"style": 0.0},
            EmotionType.HAPPY: {"style": 0.3},
            EmotionType.EXCITED: {"style": 0.5},
            EmotionType.CONFIDENT: {"style": 0.2},
            EmotionType.EMPATHETIC: {"style": 0.1},
            EmotionType.PROFESSIONAL: {"style": 0.0},
            EmotionType.SAD: {"style": 0.1}
        }
        
        return emotion_mappings.get(emotion, {"style": 0.0})
    
    def _get_audio_duration(self, audio_data: bytes, format: AudioFormat) -> int:
        """Get audio duration in milliseconds"""
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format=format.value)
            return len(audio)
        except Exception as e:
            self.logger.warning(f"Could not determine audio duration: {e}")
            return 0
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get available voices from ElevenLabs"""
        try:
            url = f"{self.base_url}/voices"
            headers = {"xi-api-key": self.api_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("voices", [])
                    else:
                        self.logger.error(f"Failed to get voices: {response.status}")
                        return []
        except Exception as e:
            self.logger.error(f"Error getting available voices: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check ElevenLabs API health"""
        try:
            url = f"{self.base_url}/user"
            headers = {"xi-api-key": self.api_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.error(f"ElevenLabs health check failed: {e}")
            return False

class AzureProvider(VoiceSynthesisProvider):
    """Azure Cognitive Services voice synthesis provider"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Azure configuration
        self.subscription_key = self.config.get('ai_services.azure.speech_key', '')
        self.region = self.config.get('ai_services.azure.speech_region', 'eastus')
        
        # Initialize speech config
        if self.subscription_key:
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.subscription_key,
                region=self.region
            )
            self.speech_config.speech_synthesis_output_format = speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
    
    async def synthesize(self, text: str, voice_settings: VoiceSettings) -> SynthesisResult:
        """Synthesize speech using Azure Cognitive Services"""
        start_time = time.time()
        
        try:
            if not self.subscription_key:
                raise Exception("Azure subscription key not configured")
            
            # Set voice
            self.speech_config.speech_synthesis_voice_name = voice_settings.voice_id
            
            # Create synthesizer
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)
            
            # Apply emotion and prosody
            ssml_text = self._create_ssml(text, voice_settings)
            
            # Synthesize
            result = synthesizer.speak_ssml_async(ssml_text).get()
            
            processing_time = int((time.time() - start_time) * 1000)
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                duration_ms = self._get_audio_duration(result.audio_data, AudioFormat.MP3)
                
                return SynthesisResult(
                    audio_data=result.audio_data,
                    format=AudioFormat.MP3,
                    duration_ms=duration_ms,
                    sample_rate=16000,  # Azure default
                    processing_time_ms=processing_time,
                    provider=VoiceProvider.AZURE,
                    voice_settings=voice_settings,
                    metadata={
                        "voice_name": voice_settings.voice_id,
                        "text_length": len(text),
                        "ssml_used": True
                    },
                    success=True
                )
            else:
                error_msg = f"Azure synthesis failed: {result.reason}"
                self.logger.error(error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            processing_time = int((time.time() - start_time) * 1000)
            self.logger.error(f"Azure synthesis failed: {e}")
            
            return SynthesisResult(
                audio_data=b"",
                format=AudioFormat.MP3,
                duration_ms=0,
                sample_rate=16000,
                processing_time_ms=processing_time,
                provider=VoiceProvider.AZURE,
                voice_settings=voice_settings,
                metadata={"error": str(e)},
                success=False,
                error_message=str(e)
            )
    
    def _create_ssml(self, text: str, voice_settings: VoiceSettings) -> str:
        """Create SSML with prosody and emotion settings"""
        # Map emotions to Azure prosody
        emotion_mappings = {
            EmotionType.HAPPY: {"rate": "medium", "pitch": "+5%", "volume": "medium"},
            EmotionType.EXCITED: {"rate": "fast", "pitch": "+10%", "volume": "loud"},
            EmotionType.CONFIDENT: {"rate": "medium", "pitch": "medium", "volume": "medium"},
            EmotionType.EMPATHETIC: {"rate": "slow", "pitch": "-5%", "volume": "soft"},
            EmotionType.PROFESSIONAL: {"rate": "medium", "pitch": "medium", "volume": "medium"},
            EmotionType.SAD: {"rate": "slow", "pitch": "-10%", "volume": "soft"},
            EmotionType.NEUTRAL: {"rate": "medium", "pitch": "medium", "volume": "medium"}
        }
        
        prosody = emotion_mappings.get(voice_settings.emotion, emotion_mappings[EmotionType.NEUTRAL])
        
        # Apply voice settings
        rate = f"{voice_settings.speaking_rate:.1f}" if voice_settings.speaking_rate != 1.0 else prosody["rate"]
        pitch_adjustment = f"{voice_settings.pitch:+.1f}%" if voice_settings.pitch != 0.0 else prosody["pitch"]
        volume = f"{voice_settings.volume:.1f}" if voice_settings.volume != 1.0 else prosody["volume"]
        
        ssml = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
            <voice name="{voice_settings.voice_id}">
                <prosody rate="{rate}" pitch="{pitch_adjustment}" volume="{volume}">
                    {text}
                </prosody>
            </voice>
        </speak>
        """
        
        return ssml.strip()
    
    def _get_audio_duration(self, audio_data: bytes, format: AudioFormat) -> int:
        """Get audio duration in milliseconds"""
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format=format.value)
            return len(audio)
        except Exception as e:
            self.logger.warning(f"Could not determine audio duration: {e}")
            return 0
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get available voices from Azure"""
        try:
            if not self.subscription_key:
                return []
            
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)
            result = synthesizer.get_voices_async().get()
            
            voices = []
            if result.reason == speechsdk.ResultReason.VoicesListRetrieved:
                for voice in result.voices:
                    voices.append({
                        "name": voice.name,
                        "display_name": voice.local_name,
                        "gender": voice.gender.name,
                        "locale": voice.locale
                    })
            
            return voices
        except Exception as e:
            self.logger.error(f"Error getting Azure voices: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check Azure service health"""
        try:
            if not self.subscription_key:
                return False
            
            # Simple synthesis test
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)
            result = synthesizer.speak_text_async("test").get()
            
            return result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted
        except Exception as e:
            self.logger.error(f"Azure health check failed: {e}")
            return False

class VoiceSynthesizer:
    """Main voice synthesis coordinator with multiple providers"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.config = get_config()
        self.session_id = session_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize providers
        self.providers = {
            VoiceProvider.ELEVENLABS: ElevenLabsProvider(),
            VoiceProvider.AZURE: AzureProvider()
        }
        
        # Default settings
        self.default_voice_settings = VoiceSettings(
            voice_id=self.config.ai_services.elevenlabs_voice_id,
            stability=0.75,
            similarity_boost=0.75,
            style=0.0,
            speaking_rate=1.0,
            pitch=0.0,
            volume=1.0,
            emotion=EmotionType.PROFESSIONAL
        )
        
        # Provider priority order
        self.provider_priority = [VoiceProvider.ELEVENLABS, VoiceProvider.AZURE]
        
        # Performance tracking
        self.metrics: List[PerformanceMetric] = []
        
        # Audio cache
        self.audio_cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    async def synthesize_speech(
        self, 
        text: str, 
        voice_settings: Optional[VoiceSettings] = None,
        preferred_provider: Optional[VoiceProvider] = None,
        use_cache: bool = True
    ) -> SynthesisResult:
        """Synthesize speech with fallback providers"""
        
        # Use default settings if none provided
        if voice_settings is None:
            voice_settings = self.default_voice_settings
        
        # Check cache first
        if use_cache:
            cache_key = self._generate_cache_key(text, voice_settings)
            cached_result = self._get_cached_audio(cache_key)
            if cached_result:
                return cached_result
        
        # Determine provider order
        providers_to_try = self._get_provider_order(preferred_provider)
        
        last_error = None
        
        for provider_type in providers_to_try:
            provider = self.providers[provider_type]
            
            try:
                # Check provider health
                if not await provider.health_check():
                    self.logger.warning(f"Provider {provider_type} health check failed, skipping")
                    continue
                
                # Synthesize speech
                result = await provider.synthesize(text, voice_settings)
                
                if result.success:
                    # Cache successful result
                    if use_cache:
                        self._cache_audio(cache_key, result)
                    
                    # Record metrics
                    await self._record_metric("synthesis_time_ms", result.processing_time_ms, "voice_synthesis")
                    await self._record_metric("audio_duration_ms", result.duration_ms, "voice_synthesis")
                    await self._record_metric("synthesis_success", 1.0, f"voice_synthesis_{provider_type}")
                    
                    self.logger.info(f"Speech synthesized successfully using {provider_type}: {result.duration_ms}ms audio in {result.processing_time_ms}ms")
                    
                    return result
                else:
                    last_error = result.error_message
                    self.logger.warning(f"Provider {provider_type} failed: {last_error}")
                    await self._record_metric("synthesis_success", 0.0, f"voice_synthesis_{provider_type}")
                    
            except Exception as e:
                last_error = str(e)
                self.logger.error(f"Provider {provider_type} error: {e}")
                await self._record_metric("synthesis_success", 0.0, f"voice_synthesis_{provider_type}")
        
        # All providers failed, return error result
        self.logger.error(f"All voice synthesis providers failed. Last error: {last_error}")
        
        return SynthesisResult(
            audio_data=b"",
            format=AudioFormat.MP3,
            duration_ms=0,
            sample_rate=22050,
            processing_time_ms=0,
            provider=VoiceProvider.ELEVENLABS,  # Default
            voice_settings=voice_settings,
            metadata={"error": "All providers failed", "last_error": last_error},
            success=False,
            error_message=f"Voice synthesis failed: {last_error}"
        )
    
    async def synthesize_with_emotion(self, text: str, emotion: EmotionType, voice_id: Optional[str] = None) -> SynthesisResult:
        """Synthesize speech with specific emotion"""
        voice_settings = VoiceSettings(
            voice_id=voice_id or self.default_voice_settings.voice_id,
            emotion=emotion,
            stability=0.75,
            similarity_boost=0.75
        )
        
        return await self.synthesize_speech(text, voice_settings)
    
    async def get_available_voices(self, provider: Optional[VoiceProvider] = None) -> Dict[VoiceProvider, List[Dict[str, Any]]]:
        """Get available voices from all or specific provider"""
        voices = {}
        
        providers_to_check = [provider] if provider else self.provider_priority
        
        for provider_type in providers_to_check:
            try:
                provider_instance = self.providers[provider_type]
                provider_voices = await provider_instance.get_available_voices()
                voices[provider_type] = provider_voices
            except Exception as e:
                self.logger.error(f"Error getting voices from {provider_type}: {e}")
                voices[provider_type] = []
        
        return voices
    
    async def test_voice_synthesis(self, test_text: str = "Hello, this is a test of the Eva Live voice synthesis system.") -> Dict[VoiceProvider, SynthesisResult]:
        """Test all voice synthesis providers"""
        results = {}
        
        for provider_type in self.provider_priority:
            try:
                result = await self.synthesize_speech(
                    test_text, 
                    preferred_provider=provider_type,
                    use_cache=False
                )
                results[provider_type] = result
            except Exception as e:
                self.logger.error(f"Test failed for {provider_type}: {e}")
                results[provider_type] = SynthesisResult(
                    audio_data=b"",
                    format=AudioFormat.MP3,
                    duration_ms=0,
                    sample_rate=22050,
                    processing_time_ms=0,
                    provider=provider_type,
                    voice_settings=self.default_voice_settings,
                    metadata={"test_error": str(e)},
                    success=False,
                    error_message=str(e)
                )
        
        return results
    
    def _get_provider_order(self, preferred_provider: Optional[VoiceProvider]) -> List[VoiceProvider]:
        """Get provider order based on preference"""
        if preferred_provider and preferred_provider in self.providers:
            # Put preferred provider first, then others
            order = [preferred_provider]
            for provider in self.provider_priority:
                if provider != preferred_provider:
                    order.append(provider)
            return order
        else:
            return self.provider_priority.copy()
    
    def _generate_cache_key(self, text: str, voice_settings: VoiceSettings) -> str:
        """Generate cache key for audio"""
        import hashlib
        
        key_data = f"{text}_{voice_settings.voice_id}_{voice_settings.emotion}_{voice_settings.speaking_rate}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_audio(self, cache_key: str) -> Optional[SynthesisResult]:
        """Get cached audio if available and not expired"""
        if cache_key in self.audio_cache:
            cached_item = self.audio_cache[cache_key]
            if time.time() - cached_item['timestamp'] < self.cache_ttl:
                cached_item['result'].metadata['cache_hit'] = True
                return cached_item['result']
            else:
                del self.audio_cache[cache_key]
        
        return None
    
    def _cache_audio(self, cache_key: str, result: SynthesisResult) -> None:
        """Cache synthesis result"""
        self.audio_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        
        # Simple cache management
        if len(self.audio_cache) > 100:
            # Remove oldest entries
            oldest_keys = sorted(
                self.audio_cache.keys(),
                key=lambda k: self.audio_cache[k]['timestamp']
            )[:20]
            
            for key in oldest_keys:
                del self.audio_cache[key]
    
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
async def create_voice_synthesizer(session_id: Optional[str] = None) -> VoiceSynthesizer:
    """Create and return a voice synthesizer instance"""
    return VoiceSynthesizer(session_id)

async def test_voice_synthesis():
    """Test function for voice synthesis"""
    try:
        # Create synthesizer
        synthesizer = VoiceSynthesizer()
        
        # Test synthesis
        result = await synthesizer.synthesize_speech("Hello! Welcome to Eva Live, your AI-powered virtual presenter.")
        
        print(f"Synthesis successful: {result.success}")
        print(f"Provider: {result.provider}")
        print(f"Duration: {result.duration_ms}ms")
        print(f"Processing time: {result.processing_time_ms}ms")
        print(f"Audio size: {len(result.audio_data)} bytes")
        
        # Test with emotion
        emotion_result = await synthesizer.synthesize_with_emotion(
            "I'm excited to show you what Eva Live can do!",
            EmotionType.EXCITED
        )
        
        print(f"Emotion synthesis successful: {emotion_result.success}")
        
        # Test all providers
        test_results = await synthesizer.test_voice_synthesis()
        print(f"Provider test results:")
        for provider, result in test_results.items():
            print(f"  {provider}: {'✓' if result.success else '✗'}")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_voice_synthesis())
