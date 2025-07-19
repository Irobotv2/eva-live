"""
Eva Live Output & Synthesis Layer

This module contains components for generating audio and visual output,
including voice synthesis, avatar rendering, and multimedia composition.
"""

from .voice_synthesis import VoiceSynthesizer, SynthesisResult, VoiceSettings, create_voice_synthesizer
from .audio_processor import AudioProcessor, AudioFormat, process_audio_async

__all__ = [
    'VoiceSynthesizer',
    'SynthesisResult', 
    'VoiceSettings',
    'create_voice_synthesizer',
    'AudioProcessor',
    'AudioFormat',
    'process_audio_async'
]
