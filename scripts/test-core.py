"""
Eva Live Core Components Test Script

This script helps test the core AI components and verify the setup.
Run this to ensure all dependencies are installed and configured correctly.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

async def main():
    """Main test function"""
    print("🚀 Eva Live Core Components Test")
    print("="*50)
    
    try:
        # Test imports
        print("Testing imports...")
        from src.core.test_integration import run_integration_tests
        
        print("✓ All imports successful")
        print()
        
        # Run integration tests
        print("Running integration tests...")
        results = await run_integration_tests()
        
        # Summary
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        if passed == total:
            print(f"\n🎉 SUCCESS: All {total} core components are working!")
            print("\nNext steps:")
            print("1. Set up your environment variables (.env file)")
            print("2. Configure API keys (OpenAI, Pinecone, etc.)")
            print("3. Start implementing the response generation system")
            return 0
        else:
            print(f"\n⚠️  {total - passed} component(s) need attention")
            print("\nThis is expected if you haven't set up API keys yet.")
            print("The core architecture is ready for development!")
            return 1
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nPlease install required dependencies:")
        print("pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
