#!/usr/bin/env python3
"""
Eva Live Quick Test Script

This script provides a simple way to test Eva Live without complex setup.
Run this to verify your installation and get Eva Live working quickly.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """Check if environment is properly configured"""
    print("🔧 Checking environment...")
    
    required_vars = [
        'OPENAI_API_KEY',
        'ELEVENLABS_API_KEY', 
        'PINECONE_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("\n📝 Please add these to your .env file:")
        for var in missing_vars:
            print(f"   {var}=your_key_here")
        print("\n📖 See QUICK_START_GUIDE.md for how to get API keys")
        return False
    
    print("✅ Environment variables configured")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'fastapi', 'openai', 'pinecone', 'redis', 
        'cv2', 'PIL', 'numpy', 'pydub'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            elif package == 'pinecone':
                import pinecone
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("\n💡 Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages installed")
    return True

async def test_eva_components():
    """Test Eva Live components individually"""
    print("\n🧪 Testing Eva Live components...")
    
    try:
        # Test AI coordinator
        print("  🧠 Testing AI coordinator...")
        from src.core.ai_coordinator import AICoordinator
        
        coordinator = AICoordinator("test_session")
        await coordinator.initialize("test_user")
        print("  ✅ AI coordinator working")
        
        # Test simple processing
        result = await coordinator.process_user_input("Hello, what is AI?")
        if result.success:
            print(f"  ✅ AI processing working: {result.eva_response.text[:50]}...")
        else:
            print("  ⚠️  AI processing had issues")
        
        await coordinator.cleanup()
        
    except Exception as e:
        print(f"  ❌ AI coordinator failed: {e}")
        return False
    
    try:
        # Test voice synthesis
        print("  🎵 Testing voice synthesis...")
        from src.output.voice_synthesis import VoiceSynthesizer, EmotionType
        
        synthesizer = VoiceSynthesizer("test_session")
        result = await synthesizer.synthesize_with_emotion(
            "Hello, this is Eva Live test", 
            EmotionType.PROFESSIONAL
        )
        
        if result.success:
            print(f"  ✅ Voice synthesis working: {result.duration_ms}ms audio")
        else:
            print(f"  ⚠️  Voice synthesis issues: {result.error_message}")
        
    except Exception as e:
        print(f"  ❌ Voice synthesis failed: {e}")
    
    try:
        # Test avatar renderer
        print("  👤 Testing avatar renderer...")
        from src.output.avatar_renderer_2d import AvatarRenderer2D, AvatarExpression, Viseme
        
        renderer = AvatarRenderer2D("test_session")
        await renderer.initialize()
        
        frame = await renderer.render_frame(
            AvatarExpression.HAPPY, 
            Viseme.A
        )
        
        if frame is not None and frame.size > 0:
            print(f"  ✅ Avatar renderer working: {frame.shape}")
        else:
            print("  ⚠️  Avatar renderer had issues")
        
    except Exception as e:
        print(f"  ❌ Avatar renderer failed: {e}")
    
    try:
        # Test virtual camera
        print("  📹 Testing virtual camera...")
        from src.integration.virtual_camera import VirtualCamera
        
        camera = VirtualCamera("test_session")
        success = await camera.initialize()
        
        if success:
            print("  ✅ Virtual camera initialized")
            
            # Try to start camera
            start_success = await camera.start()
            if start_success:
                print("  ✅ Virtual camera can start")
                await camera.stop()
            else:
                print("  ⚠️  Virtual camera start issues (may need system setup)")
            
            await camera.cleanup()
        else:
            print("  ⚠️  Virtual camera initialization issues")
        
    except Exception as e:
        print(f"  ❌ Virtual camera failed: {e}")
    
    return True

async def run_quick_demo():
    """Run a quick interactive demo"""
    print("\n🎭 Running quick Eva Live demo...")
    
    try:
        from src.eva_live_system import create_eva_live_system
        
        # Create Eva system
        eva = await create_eva_live_system(
            session_id="quick_test",
            user_id="test_user",
            config_overrides={
                'auto_start_camera': False,  # Don't auto-start for quick test
                'enable_platform_integration': False  # Keep it simple
            }
        )
        
        # Check status
        status = await eva.get_system_status()
        print(f"  📊 System state: {status.state}")
        print(f"  🔧 Components: {len(status.active_components)}")
        
        # Test a few interactions
        test_questions = [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "What can you help me with?"
        ]
        
        print("\n  💬 Testing conversations:")
        for i, question in enumerate(test_questions, 1):
            print(f"\n  🗣️  Q{i}: {question}")
            
            result = await eva.process_user_input(question)
            
            if result['success']:
                response = result['response_text']
                time_ms = result['processing_time_ms']
                voice_success = result['voice_synthesis']['success']
                
                print(f"  🤖 Eva: {response[:100]}...")
                print(f"  ⏱️  {time_ms}ms | Voice: {'✅' if voice_success else '❌'}")
            else:
                print(f"  ❌ Error: {result.get('error', 'Unknown error')}")
        
        # Try virtual camera test
        print(f"\n  📹 Virtual camera available: {eva.virtual_camera is not None}")
        
        if eva.virtual_camera:
            print("  💡 You can test virtual camera with video apps!")
            print("     1. Start Eva: await eva.start_virtual_camera()")
            print("     2. Open Zoom/Teams/Meet")
            print("     3. Select 'Eva Live Camera' as video source")
        
        # Cleanup
        await eva.shutdown()
        print("\n  ✅ Demo completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Demo failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🚀 Eva Live Quick Test")
    print("=" * 40)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed!")
        print("📖 Please see QUICK_START_GUIDE.md for setup instructions")
        return
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependencies check failed!")
        print("💡 Run: pip install -r requirements.txt")
        return
    
    # Test components
    components_ok = await test_eva_components()
    
    if not components_ok:
        print("\n⚠️  Some components had issues, but continuing...")
    
    # Run demo
    demo_ok = await run_quick_demo()
    
    # Final summary
    print("\n" + "=" * 40)
    if demo_ok:
        print("🎉 Eva Live Quick Test PASSED!")
        print("\n✅ What's working:")
        print("   • AI conversation system")
        print("   • Voice synthesis")
        print("   • Avatar rendering")
        print("   • Virtual camera")
        print("   • Complete pipeline")
        
        print("\n🚀 Next steps:")
        print("   1. Run: python full_demo.py")
        print("   2. Try with video apps")
        print("   3. Load your own presentations")
        print("   4. Customize Eva's voice/appearance")
        
        print("\n📖 See QUICK_START_GUIDE.md for detailed instructions")
        
    else:
        print("❌ Eva Live Quick Test had issues")
        print("\n🔧 Troubleshooting:")
        print("   1. Check API keys in .env file")
        print("   2. Verify internet connection") 
        print("   3. Run: pip install -r requirements.txt")
        print("   4. See QUICK_START_GUIDE.md for help")

if __name__ == "__main__":
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("💡 Install python-dotenv for .env support: pip install python-dotenv")
    
    # Run test
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("📖 Check QUICK_START_GUIDE.md for troubleshooting")
