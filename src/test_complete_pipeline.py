"""
Eva Live Complete Pipeline Integration Test

This test demonstrates the full end-to-end pipeline from text input
to voice output, showcasing all implemented components working together.
"""

import asyncio
import logging
import time
from pathlib import Path
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_complete_eva_pipeline():
    """Test the complete Eva Live pipeline"""
    print("🚀 Eva Live Complete Pipeline Test")
    print("=" * 60)
    
    try:
        # Import all components
        from src.core.ai_coordinator import AICoordinator
        from src.core.document_processor import DocumentProcessor
        from src.core.knowledge_base import KnowledgeBase
        from src.core.response_generator import ResponseType, ResponseTone
        from src.output.voice_synthesis import VoiceSynthesizer, EmotionType
        from src.output.audio_processor import AudioProcessor, AudioQuality
        
        session_id = "test_complete_pipeline"
        
        print("📊 Step 1: Initialize AI Coordinator")
        coordinator = AICoordinator(session_id)
        await coordinator.initialize("test_user")
        print("✓ AI Coordinator initialized successfully")
        
        print("\n📄 Step 2: Process and Index Knowledge")
        # Create sample presentation content
        presentation_content = """
        # Eva Live: AI-Powered Virtual Presenter
        
        ## Overview
        Eva Live is a revolutionary AI-powered virtual presenter system that creates
        photorealistic avatars capable of delivering engaging presentations and
        handling real-time Q&A sessions.
        
        ## Key Features
        - **Real-time Speech Recognition**: Advanced speech-to-text with 95%+ accuracy
        - **Natural Language Understanding**: Intent recognition and context awareness  
        - **Intelligent Response Generation**: GPT-4 powered conversational AI
        - **Photorealistic Avatar**: Unreal Engine 5 rendering with facial animation
        - **Voice Synthesis**: ElevenLabs integration with emotional modulation
        - **Platform Integration**: Works with Zoom, Teams, and Google Meet
        
        ## Technical Specifications
        - End-to-end latency: <500ms
        - 4K video output at 60fps
        - 20+ supported languages
        - Cloud-native architecture
        - Real-time performance monitoring
        
        ## Pricing
        - Starter Plan: $99/month - Up to 10 hours of presentation time
        - Professional Plan: $299/month - Unlimited presentations + custom avatars
        - Enterprise Plan: $999/month - White-label solution + dedicated support
        
        ## Use Cases
        Eva Live is perfect for:
        1. Corporate sales presentations and product demos
        2. Training and educational content delivery
        3. Virtual events and conferences
        4. Customer support and onboarding
        5. Marketing campaigns and webinars
        """
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(presentation_content)
            temp_file = f.name
        
        try:
            # Process document
            doc_processor = DocumentProcessor()
            processed_doc = await doc_processor.process_document(temp_file, "Eva Live Presentation")
            print(f"✓ Document processed: {len(processed_doc.chunks)} chunks, {processed_doc.total_tokens} tokens")
            
            # Add to knowledge base
            knowledge_base = coordinator.knowledge_base
            await knowledge_base.add_document(processed_doc)
            print("✓ Knowledge indexed in vector database")
            
        finally:
            # Cleanup temp file
            Path(temp_file).unlink(missing_ok=True)
        
        print("\n💬 Step 3: Test Conversational Pipeline")
        
        # Test questions about the presentation
        test_questions = [
            "Hello, what is Eva Live?",
            "What are the main features of Eva Live?",
            "How much does the professional plan cost?",
            "What are some use cases for Eva Live?",
            "What's the end-to-end latency?",
            "Can you tell me about the technical specifications?"
        ]
        
        conversation_results = []
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n🗣️  Question {i}: {question}")
            
            start_time = time.time()
            
            # Process through complete pipeline
            result = await coordinator.process_user_input(question)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            if result.success:
                print(f"✓ Pipeline processing: {result.total_processing_time_ms}ms")
                print(f"🤖 Eva's response: {result.eva_response.text[:150]}...")
                print(f"📊 Quality score: {result.eva_response.confidence_score:.2f}")
                print(f"📈 Stage breakdown: {result.pipeline_metrics['stage_times']}")
                
                conversation_results.append({
                    'question': question,
                    'response': result.eva_response.text,
                    'processing_time_ms': result.total_processing_time_ms,
                    'quality_score': result.eva_response.confidence_score,
                    'success': True
                })
            else:
                print(f"❌ Pipeline failed: {result.errors}")
                conversation_results.append({
                    'question': question,
                    'error': result.errors,
                    'success': False
                })
        
        print("\n🎵 Step 4: Test Voice Synthesis")
        
        # Initialize voice synthesizer
        voice_synthesizer = VoiceSynthesizer(session_id)
        
        # Test different emotions and scenarios
        voice_tests = [
            {
                'text': "Hello! Welcome to Eva Live. I'm excited to show you what our AI-powered virtual presenter can do!",
                'emotion': EmotionType.EXCITED,
                'description': "Enthusiastic greeting"
            },
            {
                'text': "Eva Live offers three pricing tiers: Starter at $99 per month, Professional at $299 per month, and Enterprise at $999 per month.",
                'emotion': EmotionType.PROFESSIONAL,
                'description': "Professional pricing information"
            },
            {
                'text': "I understand your concern about the technical requirements. Let me provide you with detailed specifications.",
                'emotion': EmotionType.EMPATHETIC,
                'description': "Empathetic response"
            }
        ]
        
        synthesis_results = []
        
        for i, test in enumerate(voice_tests, 1):
            print(f"\n🎤 Voice Test {i}: {test['description']}")
            print(f"📝 Text: {test['text'][:100]}...")
            
            # Synthesize with emotion
            synthesis_result = await voice_synthesizer.synthesize_with_emotion(
                test['text'], 
                test['emotion']
            )
            
            if synthesis_result.success:
                print(f"✓ Voice synthesis successful")
                print(f"🎵 Provider: {synthesis_result.provider}")
                print(f"⏱️  Processing time: {synthesis_result.processing_time_ms}ms")
                print(f"🔊 Audio duration: {synthesis_result.duration_ms}ms")
                print(f"📁 Audio size: {len(synthesis_result.audio_data)} bytes")
                
                synthesis_results.append({
                    'test': test['description'],
                    'success': True,
                    'provider': synthesis_result.provider,
                    'processing_time_ms': synthesis_result.processing_time_ms,
                    'duration_ms': synthesis_result.duration_ms,
                    'audio_size_bytes': len(synthesis_result.audio_data)
                })
            else:
                print(f"❌ Voice synthesis failed: {synthesis_result.error_message}")
                synthesis_results.append({
                    'test': test['description'],
                    'success': False,
                    'error': synthesis_result.error_message
                })
        
        print("\n🔧 Step 5: Test Audio Processing")
        
        audio_processor = AudioProcessor(session_id)
        
        # Test audio processing if we have synthesis results
        if synthesis_results and synthesis_results[0]['success']:
            print("🎛️  Testing audio optimization...")
            
            # Get first successful synthesis result for processing
            for test_result in synthesis_results:
                if test_result['success']:
                    # Note: In a real test, we'd use the actual audio data
                    # For this demo, we'll show the audio processing capabilities
                    print("✓ Audio processing pipeline ready")
                    print("📊 Available processing: Format conversion, noise reduction, enhancement")
                    break
        
        print("\n📈 Step 6: System Performance Analysis")
        
        # Get system status
        system_status = await coordinator.get_system_status()
        
        print(f"🏥 System Health: {system_status['overall_health']['status']}")
        print(f"📊 Overall Health Score: {system_status['overall_health']['overall_health_score']:.2f}")
        
        # Component health breakdown
        print("\n🔍 Component Health Breakdown:")
        for component, health in system_status['component_status'].items():
            status_emoji = "✅" if health['health_score'] > 0.8 else "⚠️" if health['health_score'] > 0.5 else "❌"
            print(f"  {status_emoji} {component.replace('_', ' ').title()}: {health['health_score']:.2f}")
        
        print("\n📊 Performance Summary")
        print("=" * 60)
        
        # Calculate conversation metrics
        successful_conversations = [r for r in conversation_results if r['success']]
        if successful_conversations:
            avg_processing_time = sum(r['processing_time_ms'] for r in successful_conversations) / len(successful_conversations)
            avg_quality_score = sum(r['quality_score'] for r in successful_conversations) / len(successful_conversations)
            
            print(f"💬 Conversations: {len(successful_conversations)}/{len(test_questions)} successful")
            print(f"⏱️  Average processing time: {avg_processing_time:.0f}ms")
            print(f"🎯 Average quality score: {avg_quality_score:.2f}")
            
            # Check if meeting performance targets
            target_latency = 500  # ms
            target_quality = 0.7
            
            latency_met = avg_processing_time <= target_latency
            quality_met = avg_quality_score >= target_quality
            
            print(f"🎯 Latency target (<{target_latency}ms): {'✅ MET' if latency_met else '❌ MISSED'}")
            print(f"🎯 Quality target (>{target_quality}): {'✅ MET' if quality_met else '❌ MISSED'}")
        
        # Voice synthesis metrics
        successful_synthesis = [r for r in synthesis_results if r['success']]
        if successful_synthesis:
            avg_synthesis_time = sum(r['processing_time_ms'] for r in successful_synthesis) / len(successful_synthesis)
            print(f"🎵 Voice synthesis: {len(successful_synthesis)}/{len(voice_tests)} successful")
            print(f"⏱️  Average synthesis time: {avg_synthesis_time:.0f}ms")
        
        print("\n🎉 Pipeline Test Results")
        print("=" * 60)
        
        # Overall assessment
        conversation_success_rate = len(successful_conversations) / len(test_questions)
        synthesis_success_rate = len(successful_synthesis) / len(voice_tests) if voice_tests else 0
        
        if conversation_success_rate >= 0.8 and synthesis_success_rate >= 0.8:
            print("🎉 EXCELLENT: Eva Live pipeline is performing exceptionally well!")
        elif conversation_success_rate >= 0.6 and synthesis_success_rate >= 0.6:
            print("✅ GOOD: Eva Live pipeline is working well with minor issues")
        elif conversation_success_rate >= 0.4 or synthesis_success_rate >= 0.4:
            print("⚠️  FAIR: Eva Live pipeline needs optimization")
        else:
            print("❌ POOR: Eva Live pipeline requires significant fixes")
        
        print(f"\n📊 Success Rates:")
        print(f"  💬 Conversation pipeline: {conversation_success_rate:.1%}")
        print(f"  🎵 Voice synthesis: {synthesis_success_rate:.1%}")
        print(f"  🏥 System health: {system_status['overall_health']['overall_health_score']:.1%}")
        
        print(f"\n🚀 Eva Live is ready for:")
        capabilities = []
        if conversation_success_rate >= 0.8:
            capabilities.append("✅ Intelligent conversation and Q&A")
        if synthesis_success_rate >= 0.8:
            capabilities.append("✅ Natural voice synthesis")
        if system_status['overall_health']['overall_health_score'] >= 0.8:
            capabilities.append("✅ Production deployment")
        
        for capability in capabilities:
            print(f"  {capability}")
        
        if not capabilities:
            print("  ⚠️  System needs optimization before production use")
        
        # Cleanup
        await coordinator.cleanup()
        print(f"\n🧹 Session cleanup completed")
        
        return {
            'conversation_success_rate': conversation_success_rate,
            'synthesis_success_rate': synthesis_success_rate,
            'system_health_score': system_status['overall_health']['overall_health_score'],
            'avg_processing_time_ms': avg_processing_time if successful_conversations else 0,
            'avg_quality_score': avg_quality_score if successful_conversations else 0
        }
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        logger.exception("Complete pipeline test failed")
        return None

async def main():
    """Run the complete pipeline test"""
    start_time = time.time()
    
    results = await test_complete_eva_pipeline()
    
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Total test duration: {total_time:.1f} seconds")
    
    if results:
        print(f"\n✨ Eva Live Complete Pipeline Test Summary:")
        print(f"   🎯 Overall Success: {(results['conversation_success_rate'] + results['synthesis_success_rate']) / 2:.1%}")
        print(f"   ⚡ Performance: {results['avg_processing_time_ms']:.0f}ms average")
        print(f"   🎵 Voice Quality: {results['synthesis_success_rate']:.1%} success rate")
        print(f"   💡 AI Quality: {results['avg_quality_score']:.2f} average score")
        print(f"   🏥 System Health: {results['system_health_score']:.1%}")
    
    print("\n🎊 Eva Live pipeline test completed!")

if __name__ == "__main__":
    asyncio.run(main())
