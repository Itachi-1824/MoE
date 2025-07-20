"""
Start MoE Unified Model - Single endpoint that acts like a Pollinations model
"""

import os
import sys
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Start the MoE unified model server"""
    print("🤖 Starting MoE Unified Model...")
    print("=" * 40)
    
    # Check environment
    token = os.getenv("TOKEN")
    if not token:
        print("❌ ERROR: TOKEN not set in .env file")
        sys.exit(1)
    
    print(f"✅ Token configured: {token[:8]}...")
    print(f"✅ Referrer: {os.getenv('REFERRER', 'https://pollinations.ai')}")
    
    print("\n🧠 MoE Unified Model Capabilities:")
    print("   - Intelligent task classification")
    print("   - Multi-model orchestration") 
    print("   - Text generation (26 models)")
    print("   - Image generation (4 models)")
    print("   - Audio generation (openai-audio)")
    print("   - Multimodal responses")
    print("   - Context-aware processing")
    
    print(f"\n🌐 Model will start on http://0.0.0.0:8000")
    print("📋 Available endpoints:")
    print("   - POST /openai               - OpenAI-compatible endpoint")
    print("   - GET  /{prompt}            - Simple prompt processing")
    print("   - GET  /health              - Health check")
    print()
    print("🔥 This model can return text, images, audio, or all three!")
    print("⚡ Optimized for speed - No timeouts")
    print()
    
    # Server configuration
    config = {
        "app": "moe_model_endpoint:app",
        "host": "0.0.0.0", 
        "port": 8000,
        "reload": os.getenv("ENVIRONMENT") == "development",
        "workers": 1,
        "loop": "asyncio",
        "http": "httptools", 
        "log_level": "info",
        "access_log": True,
        "use_colors": True,
        # Performance optimizations
        "backlog": 2048,
        "limit_concurrency": 1000,
        "timeout_keep_alive": 5,
        "timeout_graceful_shutdown": 30
    }
    
    # Start server
    uvicorn.run(**config)

if __name__ == "__main__":
    main()
