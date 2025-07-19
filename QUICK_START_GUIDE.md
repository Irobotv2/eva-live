# Eva Live Quick Start Guide 🚀

## How to Test and Try Eva Live

This guide will walk you through setting up and testing Eva Live step by step.

---

## 🔧 **Step 1: Environment Setup**

### **1.1 Install Dependencies**

```bash
# Clone or ensure you have the Eva Live code
# Install Python dependencies
pip install -r requirements.txt

# If requirements.txt doesn't exist, install core dependencies:
pip install fastapi uvicorn openai pinecone-client redis sentence-transformers
pip install opencv-python pillow numpy soundfile librosa noisereduce
pip install python-pptx PyPDF2 python-docx aiohttp asyncio
pip install azure-cognitiveservices-speech pydub imageio
```

### **1.2 Install System Dependencies**

#### **For Windows:**
```bash
# Install FFmpeg for video processing
# Download from: https://ffmpeg.org/download.html
# Add to PATH

# Optional: Install OBS Studio for better virtual camera support
# Download from: https://obsproject.com/
```

#### **For macOS:**
```bash
# Install FFmpeg
brew install ffmpeg

# Optional: Install OBS Studio
brew install --cask obs
```

#### **For Linux:**
```bash
# Install FFmpeg and v4l2loopback for virtual camera
sudo apt update
sudo apt install ffmpeg v4l2loopback-dkms

# Load the v4l2loopback module
sudo modprobe v4l2loopback
```

### **1.3 Configure Environment**

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
nano .env
```

**Required API Keys (add to .env):**
```env
# OpenAI (Required for AI responses)
OPENAI_API_KEY=your_openai_key_here

# ElevenLabs (Required for voice synthesis)
ELEVENLABS_API_KEY=your_elevenlabs_key_here
ELEVENLABS_VOICE_ID=your_voice_id_here

# Pinecone (Required for knowledge base)
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENVIRONMENT=your_pinecone_env

# Redis (Optional - uses local fallback if not provided)
REDIS_URL=redis://localhost:6379

# Azure Speech (Optional - fallback for voice synthesis)
AZURE_SPEECH_KEY=your_azure_key_here
AZURE_SPEECH_REGION=eastus
```

**🔑 How to Get API Keys:**
- **OpenAI**: https://platform.openai.com/api-keys
- **ElevenLabs**: https://elevenlabs.io/speech-synthesis (free tier available)
- **Pinecone**: https://www.pinecone.io/ (free tier available)
- **Azure Speech**: https://azure.microsoft.com/en-us/services/cognitive-services/speech-services/ (optional)

---

## 🧪 **Step 2: Run Basic Tests**

### **2.1 Test Core Components**

```bash
# Test 1: Core AI components
python scripts/test-core.py

# Test 2: Complete pipeline
python src/test_complete_pipeline.py

# Test 3: Individual components
python -m src.core.knowledge_base
python -m src.output.voice_synthesis
python -m src.output.avatar_renderer_2d
python -m src.integration.virtual_camera
```

### **2.2 Test Eva Live System**

```bash
# Test the complete Eva Live system
python src/eva_live_system.py
```

**Expected Output:**
```
🚀 Testing Complete Eva Live System
==================================================
✓ System initialized: ready
✓ Active components: ['ai_coordinator', 'voice_synthesizer', 'audio_processor', 'avatar_renderer', 'virtual_camera', 'platform_manager']
✓ Processing successful: True
✓ Response: Thank you for asking about Eva Live! Eva Live is a revolutionary AI-powered virtual presenter...
✓ Total time: 450ms
✓ Voice synthesis: True
✓ Avatar frames: 15
✓ Virtual camera start: True
✓ Virtual camera stopped
✓ Available platforms: ['browser']
✓ System state: ready
✓ Uptime: 5 seconds
✓ Error count: 0
✓ Total metrics collected: 25
✓ System shutdown complete

🎉 Eva Live System Test Complete!
✅ All components working together successfully!
```

---

## 🎬 **Step 3: Try Eva Live Interactive Demo**

### **3.1 Start Eva Live Server**

```bash
# Start the FastAPI server
python src/main.py

# Or use uvicorn directly
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

**Server should start on:** http://localhost:8000

### **3.2 Create a Session via API**

```bash
# Create a new Eva Live session
curl -X POST "http://localhost:8000/sessions" \
  -H "Content-Type: application/json" \
  -d '{
    "presenter_config": {
      "user_id": "test_user",
      "voice_settings": {
        "emotion": "professional"
      }
    },
    "content_config": {},
    "integration_config": {},
    "monitoring_config": {}
  }'
```

**Response:**
```json
{
  "session_id": "uuid-here",
  "status": "created",
  "endpoints": {
    "interact": "http://localhost:8000/sessions/{session_id}/interact",
    "status": "http://localhost:8000/sessions/{session_id}",
    "metrics": "http://localhost:8000/sessions/{session_id}/metrics"
  },
  "created_at": "2025-01-19T08:00:00Z",
  "expires_at": "2025-01-19T11:00:00Z"
}
```

### **3.3 Interact with Eva**

```bash
# Replace {session_id} with your actual session ID
curl -X POST "http://localhost:8000/sessions/{session_id}/interact" \
  -H "Content-Type: application/json" \
  -d '{
    "message": {
      "content": "Hello Eva, what can you do?",
      "type": "text"
    }
  }'
```

---

## 🎥 **Step 4: Test Virtual Camera**

### **4.1 Start Virtual Camera Manually**

```python
# Create and run this test script: test_virtual_camera.py

import asyncio
from src.integration.virtual_camera import VirtualCamera
from src.output.avatar_renderer_2d import AvatarRenderer2D, AvatarExpression, Viseme
import cv2
import numpy as np

async def test_virtual_camera():
    print("🎥 Testing Virtual Camera...")
    
    # Initialize components
    camera = VirtualCamera()
    avatar = AvatarRenderer2D()
    
    # Initialize
    await camera.initialize()
    await avatar.initialize()
    
    # Start camera
    success = await camera.start()
    print(f"Camera started: {success}")
    
    if success:
        print("📹 Streaming test frames for 10 seconds...")
        print("Check your video applications - Eva Live Camera should be available!")
        
        # Stream frames for 10 seconds
        for i in range(300):  # 10 seconds at 30fps
            # Render avatar frame
            expression = AvatarExpression.HAPPY if i % 60 < 30 else AvatarExpression.NEUTRAL
            viseme = Viseme.A if i % 20 < 10 else Viseme.SILENT
            
            frame = await avatar.render_frame(expression, viseme)
            
            # Add frame counter
            cv2.putText(frame, f"Eva Live Test - Frame {i}", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Send to virtual camera
            await camera.send_frame(frame)
            
            # 30fps timing
            await asyncio.sleep(1/30)
        
        print("✅ Test complete! Stopping camera...")
        await camera.stop()
    
    await camera.cleanup()
    print("🏁 Virtual camera test finished")

# Run the test
if __name__ == "__main__":
    asyncio.run(test_virtual_camera())
```

```bash
# Run the virtual camera test
python test_virtual_camera.py
```

### **4.2 Test in Video Applications**

1. **Open Zoom/Teams/Google Meet**
2. **Go to video settings**
3. **Look for "Eva Live Camera" in camera list**
4. **Select it as your camera source**
5. **You should see Eva's avatar with test frames!**

---

## 🎤 **Step 5: Test Voice Synthesis**

### **5.1 Test Voice Generation**

```python
# Create test_voice.py

import asyncio
from src.output.voice_synthesis import VoiceSynthesizer, EmotionType
import tempfile
import pygame

async def test_voice():
    print("🎤 Testing Voice Synthesis...")
    
    synthesizer = VoiceSynthesizer()
    
    # Test different emotions
    test_phrases = [
        ("Hello! Welcome to Eva Live!", EmotionType.EXCITED),
        ("I understand your question.", EmotionType.PROFESSIONAL),
        ("Let me help you with that.", EmotionType.EMPATHETIC),
        ("This is a demonstration of Eva's voice capabilities.", EmotionType.CONFIDENT)
    ]
    
    for phrase, emotion in test_phrases:
        print(f"🗣️  Synthesizing: '{phrase}' with {emotion.value} emotion")
        
        result = await synthesizer.synthesize_with_emotion(phrase, emotion)
        
        if result.success:
            print(f"✅ Success! Duration: {result.duration_ms}ms, Provider: {result.provider}")
            
            # Save audio file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                f.write(result.audio_data)
                print(f"📁 Audio saved to: {f.name}")
                
                # Try to play audio (requires pygame)
                try:
                    pygame.mixer.init()
                    pygame.mixer.music.load(f.name)
                    pygame.mixer.music.play()
                    
                    # Wait for playback to finish
                    while pygame.mixer.music.get_busy():
                        await asyncio.sleep(0.1)
                    
                    print("🔊 Playback complete!")
                except Exception as e:
                    print(f"⚠️  Could not play audio: {e}")
                    print("You can manually play the saved file")
        else:
            print(f"❌ Failed: {result.error_message}")
        
        print()

if __name__ == "__main__":
    asyncio.run(test_voice())
```

```bash
# Install pygame for audio playback
pip install pygame

# Run voice test
python test_voice.py
```

---

## 🎯 **Step 6: Complete Integration Test**

### **6.1 Full Eva Live Demo**

```python
# Create full_demo.py

import asyncio
from src.eva_live_system import create_eva_live_system
import tempfile
from pathlib import Path

async def full_eva_demo():
    print("🎭 Eva Live Complete Demo")
    print("=" * 50)
    
    # Create Eva Live system
    print("🚀 Initializing Eva Live...")
    eva = await create_eva_live_system(
        session_id="demo_session",
        user_id="demo_user",
        config_overrides={
            'auto_start_camera': True,  # Automatically start virtual camera
            'enable_platform_integration': True
        }
    )
    
    print("✅ Eva Live initialized!")
    
    # Check system status
    status = await eva.get_system_status()
    print(f"📊 System state: {status.state}")
    print(f"🔧 Active components: {len(status.active_components)}")
    
    # Create sample presentation content
    print("\n📄 Loading sample presentation...")
    sample_content = """
    # Eva Live Demo Presentation
    
    ## What is Eva Live?
    Eva Live is an AI-powered virtual presenter that can:
    - Understand presentation content
    - Answer questions intelligently  
    - Speak with natural voice synthesis
    - Display facial expressions and lip-sync
    - Stream to video conferencing platforms
    
    ## Key Features
    - Real-time AI processing
    - Natural language understanding
    - Voice synthesis with emotions
    - 2D avatar with expressions
    - Virtual camera integration
    - Multi-platform support
    
    ## Use Cases
    - Corporate presentations
    - Virtual events
    - Customer support
    - Educational content
    - Training sessions
    """
    
    # Save and load presentation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(sample_content)
        temp_file = f.name
    
    success = await eva.load_presentation(temp_file, "Eva Live Demo")
    print(f"📚 Presentation loaded: {success}")
    
    # Cleanup temp file
    Path(temp_file).unlink()
    
    # Test interactions
    print("\n💬 Testing Eva's responses...")
    
    questions = [
        "What is Eva Live?",
        "What are the key features?", 
        "How can Eva Live be used?",
        "What makes Eva special?",
        "Can you tell me about the voice synthesis?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n🗣️  Question {i}: {question}")
        
        result = await eva.process_user_input(question)
        
        if result['success']:
            print(f"🤖 Eva: {result['response_text'][:150]}...")
            print(f"⏱️  Processing time: {result['processing_time_ms']}ms")
            print(f"🎵 Voice: {result['voice_synthesis']['success']} ({result['voice_synthesis']['duration_ms']}ms)")
            print(f"👤 Avatar frames: {result['avatar_animation']['frames_generated']}")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    # Check if virtual camera is running
    print(f"\n📹 Virtual camera running: {eva.virtual_camera.is_running() if eva.virtual_camera else False}")
    
    if eva.virtual_camera and eva.virtual_camera.is_running():
        print("🎥 Eva Live Camera is streaming!")
        print("👀 Check your video applications - Eva should be available as a camera source!")
        
        # Keep streaming for a bit
        print("⏳ Streaming for 10 more seconds...")
        await asyncio.sleep(10)
    
    # Get final metrics
    print("\n📊 Performance Summary:")
    metrics = eva.get_metrics()
    
    # Group metrics by component
    metric_summary = {}
    for metric in metrics:
        component = metric.component
        if component not in metric_summary:
            metric_summary[component] = []
        metric_summary[component].append(metric)
    
    for component, component_metrics in metric_summary.items():
        avg_time = sum(m.metric_value for m in component_metrics if 'time' in m.metric_type) / max(1, len([m for m in component_metrics if 'time' in m.metric_type]))
        print(f"  📈 {component}: {len(component_metrics)} metrics, avg time: {avg_time:.1f}ms")
    
    # Cleanup
    print("\n🧹 Shutting down Eva Live...")
    await eva.shutdown()
    
    print("\n🎉 Demo Complete!")
    print("🎊 Eva Live is ready for production use!")

if __name__ == "__main__":
    asyncio.run(full_eva_demo())
```

```bash
# Run the complete demo
python full_demo.py
```

---

## 🎯 **Step 7: Try with Video Conferencing**

### **7.1 Test with Real Video Apps**

1. **Start Eva Live:**
   ```bash
   python full_demo.py
   ```

2. **Open any video app:**
   - Zoom
   - Microsoft Teams  
   - Google Meet
   - Discord
   - OBS Studio
   - Any webcam application

3. **Select Camera:**
   - Go to video settings
   - Choose "Eva Live Camera"
   - You should see Eva's avatar!

4. **Test in a Meeting:**
   ```python
   # Add this to your demo script
   await eva.join_meeting("https://meet.google.com/your-meeting-link")
   ```

### **7.2 Browser-Based Testing**

```python
# Test browser integration
import webbrowser

# This will open the meeting in your browser
# Eva will be available as "Eva Live Camera"
webbrowser.open("https://meet.google.com/new")
```

---

## 🚨 **Troubleshooting**

### **Common Issues:**

#### **1. Virtual Camera Not Appearing**

**Windows:**
```bash
# Install OBS Studio for better virtual camera support
# Or ensure FFmpeg is in PATH
```

**macOS:**
```bash
# Install obs-mac-virtualcam plugin
# Or give terminal camera permissions in System Preferences
```

**Linux:**
```bash
# Ensure v4l2loopback is loaded
sudo modprobe v4l2loopback
lsmod | grep v4l2loopback
```

#### **2. Voice Synthesis Not Working**

```bash
# Check API keys in .env
echo $ELEVENLABS_API_KEY

# Test API connection
curl -H "xi-api-key: YOUR_API_KEY" https://api.elevenlabs.io/v1/voices
```

#### **3. AI Responses Failing**

```bash
# Check OpenAI API key
echo $OPENAI_API_KEY

# Test OpenAI connection
curl -H "Authorization: Bearer YOUR_API_KEY" https://api.openai.com/v1/models
```

#### **4. Dependencies Missing**

```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt

# Install system dependencies
# See Step 1.2 for your platform
```

---

## 🎊 **You're Ready!**

After following this guide, you should have:

✅ **Eva Live running locally**  
✅ **Virtual camera streaming**  
✅ **Voice synthesis working**  
✅ **AI responses functioning**  
✅ **Integration with video apps**  

**Next steps:**
- Try different presentation content
- Experiment with different questions
- Test in real video meetings
- Customize Eva's voice and appearance
- Integrate with your own applications

**Need help?** Check the logs in the console output for detailed error messages and troubleshooting information.

**🎉 Welcome to the future of AI virtual presenting!**
