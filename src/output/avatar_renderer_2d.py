"""
Eva Live 2D Avatar Renderer

This module handles 2D avatar rendering with facial expressions, lip-sync,
and real-time video composition for Eva Live presentations.
"""

import asyncio
import logging
import time
import io
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json

# Image and video processing
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio

from ..shared.config import get_config
from ..shared.models import PerformanceMetric
from .voice_synthesis import EmotionType

class AvatarExpression(str, Enum):
    """Available avatar expressions"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    EXCITED = "excited"
    SURPRISED = "surprised"
    CONFIDENT = "confident"
    THOUGHTFUL = "thoughtful"
    EMPATHETIC = "empathetic"
    SPEAKING = "speaking"
    LISTENING = "listening"

class Viseme(str, Enum):
    """Phoneme-based mouth shapes for lip-sync"""
    SILENT = "silent"           # Mouth closed
    A = "a"                     # Open mouth (ah, apple)
    E = "e"                     # Medium open (eh, elephant) 
    I = "i"                     # Small opening (ee, eagle)
    O = "o"                     # Round mouth (oh, orange)
    U = "u"                     # Pursed lips (oo, unicorn)
    M_B_P = "mbp"              # Closed lips (m, b, p sounds)
    F_V = "fv"                 # Teeth on lip (f, v sounds)
    TH = "th"                  # Tongue between teeth
    T_D_N_L = "tdnl"           # Tongue to roof (t, d, n, l)
    S_Z = "sz"                 # Narrow opening (s, z sounds)
    R = "r"                    # Rounded opening (r sound)

@dataclass
class AvatarFrame:
    """Single frame of avatar animation"""
    expression: AvatarExpression
    viseme: Viseme
    eye_blink: bool = False
    eye_direction: Tuple[float, float] = (0.0, 0.0)  # x, y offset from center
    timestamp_ms: int = 0

@dataclass
class AvatarAssets:
    """Avatar image assets and configurations"""
    base_images: Dict[AvatarExpression, str]  # File paths to expression images
    mouth_overlays: Dict[Viseme, str]         # File paths to mouth shapes
    eye_overlays: Dict[str, str]              # Eye states (open, closed, etc.)
    avatar_config: Dict[str, Any]             # Position and size configs

class AvatarRenderer2D:
    """2D Avatar rendering engine"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.config = get_config()
        self.session_id = session_id
        self.logger = logging.getLogger(__name__)
        
        # Rendering settings
        self.output_width = 1920
        self.output_height = 1080
        self.fps = 30
        self.avatar_scale = 1.0
        
        # Animation state
        self.current_expression = AvatarExpression.NEUTRAL
        self.current_viseme = Viseme.SILENT
        self.blink_timer = 0
        self.next_blink = 0
        
        # Avatar assets
        self.assets: Optional[AvatarAssets] = None
        self.loaded_images: Dict[str, np.ndarray] = {}
        
        # Performance tracking
        self.metrics: List[PerformanceMetric] = []
        
        # Background and composition
        self.background_image: Optional[np.ndarray] = None
        self.avatar_position = (960, 540)  # Center of 1920x1080
        
    async def initialize(self, assets_path: Optional[str] = None) -> None:
        """Initialize avatar renderer with assets"""
        try:
            if assets_path is None:
                assets_path = self.config.get('avatar.assets_path', 'assets/avatar')
            
            # Load avatar configuration
            await self._load_avatar_assets(assets_path)
            
            # Load default background
            await self._load_background()
            
            # Initialize OpenCV for video processing
            self._initialize_video_processing()
            
            self.logger.info("2D Avatar renderer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize avatar renderer: {e}")
            raise
    
    async def render_frame(
        self,
        expression: AvatarExpression = AvatarExpression.NEUTRAL,
        viseme: Viseme = Viseme.SILENT,
        emotion_intensity: float = 1.0,
        timestamp_ms: int = 0
    ) -> np.ndarray:
        """Render a single avatar frame"""
        start_time = time.time()
        
        try:
            # Create base frame with background
            frame = self.background_image.copy() if self.background_image is not None else self._create_default_background()
            
            # Load and position avatar base image
            avatar_base = await self._get_expression_image(expression, emotion_intensity)
            
            # Apply mouth shape for lip-sync
            avatar_with_mouth = await self._apply_mouth_shape(avatar_base, viseme)
            
            # Apply eye animation (blinking, movement)
            avatar_with_eyes = await self._apply_eye_animation(avatar_with_mouth, timestamp_ms)
            
            # Composite avatar onto background
            final_frame = await self._composite_avatar(frame, avatar_with_eyes)
            
            # Add any overlays (subtitles, indicators, etc.)
            final_frame = await self._add_overlays(final_frame, timestamp_ms)
            
            # Update state
            self.current_expression = expression
            self.current_viseme = viseme
            
            # Record performance
            render_time = int((time.time() - start_time) * 1000)
            await self._record_metric("frame_render_time_ms", render_time, "avatar_renderer_2d")
            
            return final_frame
            
        except Exception as e:
            self.logger.error(f"Error rendering avatar frame: {e}")
            # Return blank frame on error
            return self._create_error_frame()
    
    async def render_sequence(
        self,
        frames: List[AvatarFrame],
        output_path: Optional[str] = None
    ) -> List[np.ndarray]:
        """Render a sequence of avatar frames"""
        try:
            rendered_frames = []
            
            for i, frame_data in enumerate(frames):
                frame = await self.render_frame(
                    frame_data.expression,
                    frame_data.viseme,
                    timestamp_ms=frame_data.timestamp_ms
                )
                rendered_frames.append(frame)
                
                # Log progress for long sequences
                if len(frames) > 10 and i % 10 == 0:
                    self.logger.debug(f"Rendered {i+1}/{len(frames)} frames")
            
            # Save as video if output path provided
            if output_path:
                await self._save_video_sequence(rendered_frames, output_path)
            
            self.logger.info(f"Rendered {len(frames)} avatar frames")
            return rendered_frames
            
        except Exception as e:
            self.logger.error(f"Error rendering avatar sequence: {e}")
            return []
    
    def emotion_to_expression(self, emotion: EmotionType, intensity: float = 1.0) -> AvatarExpression:
        """Convert emotion type to avatar expression"""
        emotion_mapping = {
            EmotionType.NEUTRAL: AvatarExpression.NEUTRAL,
            EmotionType.HAPPY: AvatarExpression.HAPPY,
            EmotionType.SAD: AvatarExpression.SAD,
            EmotionType.EXCITED: AvatarExpression.EXCITED,
            EmotionType.CONFIDENT: AvatarExpression.CONFIDENT,
            EmotionType.EMPATHETIC: AvatarExpression.EMPATHETIC,
            EmotionType.PROFESSIONAL: AvatarExpression.CONFIDENT
        }
        
        return emotion_mapping.get(emotion, AvatarExpression.NEUTRAL)
    
    def text_to_visemes(self, text: str, timing_ms: List[int]) -> List[Tuple[Viseme, int]]:
        """Convert text to sequence of visemes with timing"""
        try:
            # Simple phoneme-to-viseme mapping
            # In production, this would use more sophisticated phoneme analysis
            
            viseme_sequence = []
            words = text.lower().split()
            
            if not timing_ms or len(timing_ms) < len(words):
                # Generate even timing if not provided
                total_duration = timing_ms[-1] if timing_ms else len(text) * 100
                timing_ms = [int(i * total_duration / len(words)) for i in range(len(words) + 1)]
            
            for i, word in enumerate(words):
                word_start = timing_ms[i]
                word_end = timing_ms[i + 1] if i + 1 < len(timing_ms) else word_start + 500
                word_duration = word_end - word_start
                
                # Analyze word for dominant sounds
                visemes = self._word_to_visemes(word)
                
                # Distribute visemes across word duration
                if visemes:
                    viseme_duration = word_duration // len(visemes)
                    for j, viseme in enumerate(visemes):
                        timestamp = word_start + (j * viseme_duration)
                        viseme_sequence.append((viseme, timestamp))
                
                # Add silent pause between words
                if i < len(words) - 1:
                    viseme_sequence.append((Viseme.SILENT, word_end))
            
            return viseme_sequence
            
        except Exception as e:
            self.logger.error(f"Error converting text to visemes: {e}")
            return [(Viseme.SILENT, 0)]
    
    def _word_to_visemes(self, word: str) -> List[Viseme]:
        """Convert a single word to viseme sequence"""
        # Simplified phoneme mapping
        # In production, this would use a proper phoneme dictionary
        
        visemes = []
        
        # Basic letter-to-viseme mapping
        for char in word:
            if char in 'aeiou':
                if char == 'a':
                    visemes.append(Viseme.A)
                elif char == 'e':
                    visemes.append(Viseme.E)
                elif char == 'i':
                    visemes.append(Viseme.I)
                elif char == 'o':
                    visemes.append(Viseme.O)
                elif char == 'u':
                    visemes.append(Viseme.U)
            elif char in 'mbp':
                visemes.append(Viseme.M_B_P)
            elif char in 'fv':
                visemes.append(Viseme.F_V)
            elif char in 'tdnl':
                visemes.append(Viseme.T_D_N_L)
            elif char in 'sz':
                visemes.append(Viseme.S_Z)
            elif char == 'r':
                visemes.append(Viseme.R)
            elif char in 'th':
                visemes.append(Viseme.TH)
        
        # Default to speaking if no specific visemes found
        if not visemes:
            visemes = [Viseme.A, Viseme.SILENT]
        
        return visemes
    
    async def _load_avatar_assets(self, assets_path: str) -> None:
        """Load avatar image assets from directory"""
        try:
            assets_dir = Path(assets_path)
            
            if not assets_dir.exists():
                # Create default assets if directory doesn't exist
                await self._create_default_assets(assets_dir)
            
            # Load configuration
            config_file = assets_dir / "avatar_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    avatar_config = json.load(f)
            else:
                avatar_config = self._get_default_avatar_config()
            
            # Define asset paths
            base_images = {}
            mouth_overlays = {}
            eye_overlays = {}
            
            for expression in AvatarExpression:
                img_path = assets_dir / "expressions" / f"{expression.value}.png"
                if img_path.exists():
                    base_images[expression] = str(img_path)
            
            for viseme in Viseme:
                mouth_path = assets_dir / "mouths" / f"{viseme.value}.png"
                if mouth_path.exists():
                    mouth_overlays[viseme] = str(mouth_path)
            
            # Eye states
            for eye_state in ["open", "closed", "half"]:
                eye_path = assets_dir / "eyes" / f"{eye_state}.png"
                if eye_path.exists():
                    eye_overlays[eye_state] = str(eye_path)
            
            self.assets = AvatarAssets(
                base_images=base_images,
                mouth_overlays=mouth_overlays,
                eye_overlays=eye_overlays,
                avatar_config=avatar_config
            )
            
            # Pre-load critical images
            await self._preload_images()
            
            self.logger.info(f"Loaded avatar assets from {assets_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading avatar assets: {e}")
            # Create minimal default assets
            self.assets = self._create_minimal_assets()
    
    async def _create_default_assets(self, assets_dir: Path) -> None:
        """Create default avatar assets using generated images"""
        try:
            assets_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (assets_dir / "expressions").mkdir(exist_ok=True)
            (assets_dir / "mouths").mkdir(exist_ok=True)
            (assets_dir / "eyes").mkdir(exist_ok=True)
            (assets_dir / "backgrounds").mkdir(exist_ok=True)
            
            # Generate simple avatar images using PIL
            await self._generate_simple_avatar_images(assets_dir)
            
            # Save configuration
            config = self._get_default_avatar_config()
            with open(assets_dir / "avatar_config.json", 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"Created default avatar assets in {assets_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating default assets: {e}")
    
    async def _generate_simple_avatar_images(self, assets_dir: Path) -> None:
        """Generate simple avatar images using PIL"""
        try:
            # Avatar dimensions
            avatar_width, avatar_height = 400, 600
            
            # Generate expression images
            for expression in AvatarExpression:
                img = Image.new('RGBA', (avatar_width, avatar_height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                
                # Draw simple avatar shape (oval for head)
                head_color = (255, 220, 177)  # Skin tone
                draw.ellipse([100, 50, 300, 300], fill=head_color, outline=(200, 180, 140))
                
                # Draw eyes based on expression
                eye_y = 150
                if expression == AvatarExpression.HAPPY:
                    # Happy eyes (curved)
                    draw.arc([130, eye_y-10, 160, eye_y+10], 0, 180, fill=(0, 0, 0), width=3)
                    draw.arc([240, eye_y-10, 270, eye_y+10], 0, 180, fill=(0, 0, 0), width=3)
                elif expression == AvatarExpression.SAD:
                    # Sad eyes (droopy)
                    draw.arc([130, eye_y-5, 160, eye_y+15], 180, 360, fill=(0, 0, 0), width=3)
                    draw.arc([240, eye_y-5, 270, eye_y+15], 180, 360, fill=(0, 0, 0), width=3)
                else:
                    # Normal eyes
                    draw.ellipse([135, eye_y, 155, eye_y+15], fill=(0, 0, 0))
                    draw.ellipse([245, eye_y, 265, eye_y+15], fill=(0, 0, 0))
                
                # Draw mouth based on expression
                mouth_y = 220
                if expression == AvatarExpression.HAPPY:
                    draw.arc([170, mouth_y, 230, mouth_y+30], 0, 180, fill=(0, 0, 0), width=3)
                elif expression == AvatarExpression.SAD:
                    draw.arc([170, mouth_y-10, 230, mouth_y+20], 180, 360, fill=(0, 0, 0), width=3)
                else:
                    draw.line([180, mouth_y+10, 220, mouth_y+10], fill=(0, 0, 0), width=3)
                
                # Save expression image
                img.save(assets_dir / "expressions" / f"{expression.value}.png")
            
            # Generate mouth shapes
            for viseme in Viseme:
                img = Image.new('RGBA', (60, 40), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                
                if viseme == Viseme.A:
                    draw.ellipse([10, 5, 50, 35], fill=(100, 50, 50), outline=(0, 0, 0))
                elif viseme == Viseme.E:
                    draw.ellipse([15, 10, 45, 30], fill=(100, 50, 50), outline=(0, 0, 0))
                elif viseme == Viseme.I:
                    draw.ellipse([20, 15, 40, 25], fill=(100, 50, 50), outline=(0, 0, 0))
                elif viseme == Viseme.O:
                    draw.ellipse([15, 8, 45, 32], fill=(100, 50, 50), outline=(0, 0, 0))
                elif viseme == Viseme.U:
                    draw.ellipse([18, 12, 42, 28], fill=(100, 50, 50), outline=(0, 0, 0))
                elif viseme == Viseme.M_B_P:
                    draw.line([20, 20, 40, 20], fill=(0, 0, 0), width=3)
                else:
                    # Default small opening
                    draw.ellipse([25, 18, 35, 22], fill=(100, 50, 50), outline=(0, 0, 0))
                
                img.save(assets_dir / "mouths" / f"{viseme.value}.png")
            
            # Generate eye states
            for eye_state in ["open", "closed", "half"]:
                img = Image.new('RGBA', (30, 20), (0, 0, 0, 0))
                draw = ImageDraw.Draw(img)
                
                if eye_state == "open":
                    draw.ellipse([5, 5, 25, 15], fill=(0, 0, 0))
                elif eye_state == "closed":
                    draw.line([5, 10, 25, 10], fill=(0, 0, 0), width=2)
                else:  # half
                    draw.arc([5, 8, 25, 12], 0, 180, fill=(0, 0, 0), width=2)
                
                img.save(assets_dir / "eyes" / f"{eye_state}.png")
            
            self.logger.info("Generated simple avatar images")
            
        except Exception as e:
            self.logger.error(f"Error generating avatar images: {e}")
    
    def _get_default_avatar_config(self) -> Dict[str, Any]:
        """Get default avatar configuration"""
        return {
            "avatar_size": [400, 600],
            "avatar_position": [760, 240],  # Position in 1920x1080 frame
            "mouth_position": [200, 220],   # Position relative to avatar
            "eye_positions": [[145, 150], [255, 150]],  # Left and right eye positions
            "blink_interval_ms": [2000, 5000],  # Random blink interval range
            "expression_transition_ms": 500,     # Time to blend between expressions
            "mouth_sync_sensitivity": 0.8        # Lip-sync responsiveness
        }
    
    def _create_minimal_assets(self) -> AvatarAssets:
        """Create minimal assets for fallback"""
        return AvatarAssets(
            base_images={},
            mouth_overlays={},
            eye_overlays={},
            avatar_config=self._get_default_avatar_config()
        )
    
    async def _preload_images(self) -> None:
        """Pre-load critical images into memory"""
        if not self.assets:
            return
        
        # Load most common expressions
        critical_expressions = [
            AvatarExpression.NEUTRAL,
            AvatarExpression.SPEAKING,
            AvatarExpression.HAPPY
        ]
        
        for expression in critical_expressions:
            if expression in self.assets.base_images:
                await self._load_image(self.assets.base_images[expression])
    
    async def _load_image(self, image_path: str) -> np.ndarray:
        """Load and cache an image"""
        if image_path in self.loaded_images:
            return self.loaded_images[image_path]
        
        try:
            # Load with PIL then convert to OpenCV format
            pil_image = Image.open(image_path).convert('RGBA')
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)
            
            self.loaded_images[image_path] = cv_image
            return cv_image
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {e}")
            return self._create_placeholder_image()
    
    def _create_placeholder_image(self) -> np.ndarray:
        """Create a placeholder image when assets fail to load"""
        img = np.zeros((600, 400, 4), dtype=np.uint8)
        img[:, :, 3] = 255  # Alpha channel
        
        # Draw simple placeholder
        cv2.rectangle(img, (50, 50), (350, 550), (100, 100, 100, 255), -1)
        cv2.putText(img, "Avatar", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255, 255), 3)
        
        return img
    
    async def _get_expression_image(self, expression: AvatarExpression, intensity: float) -> np.ndarray:
        """Get avatar image for specific expression"""
        if not self.assets or expression not in self.assets.base_images:
            return self._create_placeholder_image()
        
        image_path = self.assets.base_images[expression]
        base_image = await self._load_image(image_path)
        
        # Apply intensity modification if needed
        if intensity != 1.0:
            base_image = self._apply_expression_intensity(base_image, intensity)
        
        return base_image
    
    def _apply_expression_intensity(self, image: np.ndarray, intensity: float) -> np.ndarray:
        """Apply expression intensity modification"""
        # Simple intensity application by blending with neutral
        if intensity >= 1.0:
            return image
        
        # Blend towards neutral expression
        neutral_image = self._create_placeholder_image()  # Would use actual neutral in production
        
        blended = cv2.addWeighted(image, intensity, neutral_image, 1.0 - intensity, 0)
        return blended
    
    async def _apply_mouth_shape(self, avatar_image: np.ndarray, viseme: Viseme) -> np.ndarray:
        """Apply mouth shape overlay for lip-sync"""
        try:
            if not self.assets or viseme not in self.assets.mouth_overlays:
                return avatar_image
            
            mouth_path = self.assets.mouth_overlays[viseme]
            mouth_overlay = await self._load_image(mouth_path)
            
            # Get mouth position from config
            mouth_pos = self.assets.avatar_config.get("mouth_position", [200, 220])
            
            # Composite mouth onto avatar
            result = avatar_image.copy()
            self._overlay_image(result, mouth_overlay, mouth_pos[0], mouth_pos[1])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error applying mouth shape: {e}")
            return avatar_image
    
    async def _apply_eye_animation(self, avatar_image: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """Apply eye animation (blinking, movement)"""
        try:
            # Simple blink animation
            blink_interval = self.assets.avatar_config.get("blink_interval_ms", [2000, 5000])
            
            # Check if it's time to blink
            if timestamp_ms - self.blink_timer > self.next_blink:
                self.blink_timer = timestamp_ms
                self.next_blink = np.random.randint(blink_interval[0], blink_interval[1])
                
                # Apply blink for short duration
                if (timestamp_ms - self.blink_timer) < 150:  # 150ms blink
                    return self._apply_blink(avatar_image)
            
            return avatar_image
            
        except Exception as e:
            self.logger.error(f"Error applying eye animation: {e}")
            return avatar_image
    
    def _apply_blink(self, avatar_image: np.ndarray) -> np.ndarray:
        """Apply blink effect to avatar"""
        # Simple implementation - would overlay closed eyes in production
        return avatar_image
    
    async def _composite_avatar(self, background: np.ndarray, avatar: np.ndarray) -> np.ndarray:
        """Composite avatar onto background"""
        try:
            result = background.copy()
            
            # Get avatar position
            pos = self.assets.avatar_config.get("avatar_position", [760, 240]) if self.assets else [760, 240]
            
            # Scale avatar if needed
            if self.avatar_scale != 1.0:
                new_size = (int(avatar.shape[1] * self.avatar_scale), int(avatar.shape[0] * self.avatar_scale))
                avatar = cv2.resize(avatar, new_size)
            
            # Composite with alpha blending
            self._overlay_image(result, avatar, pos[0], pos[1])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error compositing avatar: {e}")
            return background
    
    def _overlay_image(self, background: np.ndarray, overlay: np.ndarray, x: int, y: int) -> None:
        """Overlay image with alpha blending"""
        try:
            # Ensure overlay has alpha channel
            if overlay.shape[2] == 3:
                overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
            
            # Calculate overlay region
            h, w = overlay.shape[:2]
            
            # Clip to background bounds
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(background.shape[1], x + w), min(background.shape[0], y + h)
            
            if x2 <= x1 or y2 <= y1:
                return
            
            # Extract regions
            bg_region = background[y1:y2, x1:x2]
            overlay_region = overlay[y1-y:y2-y, x1-x:x2-x]
            
            # Alpha blend
            alpha = overlay_region[:, :, 3:4] / 255.0
            
            # Blend
            blended = bg_region * (1 - alpha) + overlay_region[:, :, :3] * alpha
            background[y1:y2, x1:x2] = blended.astype(np.uint8)
            
        except Exception as e:
            self.logger.error(f"Error overlaying image: {e}")
    
    async def _add_overlays(self, frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
        """Add any additional overlays (subtitles, indicators, etc.)"""
        # Placeholder for subtitle/indicator overlays
        return frame
    
    def _create_default_background(self) -> np.ndarray:
        """Create a default background"""
        # Create gradient background
        background = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
        
        # Create blue gradient
        for y in range(self.output_height):
            intensity = int(255 * (1 - y / self.output_height) * 0.3)
            background[y, :] = [intensity + 50, intensity + 50, intensity + 100]
        
        return background
    
    def _create_error_frame(self) -> np.ndarray:
        """Create error frame when rendering fails"""
        frame = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
        cv2.putText(frame, "Avatar Rendering Error", (600, 540), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        return frame
    
    async def _load_background(self) -> None:
        """Load background image"""
        try:
            # Try to load configured background
            bg_path = self.config.get('avatar.background_path')
            if bg_path and Path(bg_path).exists():
                self.background_image = cv2.imread(bg_path)
                self.background_image = cv2.resize(self.background_image, (self.output_width, self.output_height))
            else:
                self.background_image = self._create_default_background()
                
        except Exception as e:
            self.logger.error(f"Error loading background: {e}")
            self.background_image = self._create_default_background()
    
    def _initialize_video_processing(self) -> None:
        """Initialize video processing settings"""
        # Set OpenCV optimization flags
        cv2.setUseOptimized(True)
        cv2.setNumThreads(4)
    
    async def _save_video_sequence(self, frames: List[np.ndarray], output_path: str) -> None:
        """Save frame sequence as video"""
        try:
            if not frames:
                return
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, 
                                   (self.output_width, self.output_height))
            
            for frame in frames:
                writer.write(frame)
            
            writer.release()
            self.logger.info(f"Saved video sequence to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving video sequence: {e}")
    
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
async def create_avatar_renderer(session_id: Optional[str] = None) -> AvatarRenderer2D:
    """Create and initialize avatar renderer"""
    renderer = AvatarRenderer2D(session_id)
    await renderer.initialize()
    return renderer

async def test_avatar_renderer():
    """Test avatar renderer functionality"""
    try:
        # Initialize renderer
        renderer = AvatarRenderer2D()
        await renderer.initialize()
        
        # Test single frame rendering
        frame = await renderer.render_frame(
            AvatarExpression.HAPPY,
            Viseme.A,
            emotion_intensity=1.0
        )
        
        print(f"✓ Rendered frame: {frame.shape}")
        
        # Test viseme conversion
        visemes = renderer.text_to_visemes("Hello world", [0, 500, 1000])
        print(f"✓ Generated {len(visemes)} visemes")
        
        # Test emotion mapping
        expression = renderer.emotion_to_expression(EmotionType.EXCITED)
        print(f"✓ Mapped emotion to expression: {expression}")
        
        print("Avatar renderer test completed successfully!")
        
    except Exception as e:
        print(f"Avatar renderer test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_avatar_renderer())
