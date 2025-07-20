"""
MoE Unified Model Endpoint - Single endpoint that acts like a Pollinations model
but uses MoE intelligence to orchestrate multiple models for enhanced responses
"""

import asyncio
import json
import base64
import time
from typing import Dict, List, Optional, Any, Union
import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from models_registry import registry
from moe_orchestrator import orchestrator
from pollinations_client import client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app - single model endpoint
app = FastAPI(
    title="MoE Unified Model",
    description="Context-aware MoE model that intelligently combines multiple AI models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MultimodalResponse:
    """Unified response format that can contain text, images, and audio"""
    def __init__(self):
        self.text_content = ""
        self.images = []
        self.audio = []
        self.metadata = {}
        self.models_used = []
        self.strategy = ""
        self.latency = 0.0
    
    def add_text(self, content: str):
        """Add text content to the response"""
        if self.text_content:
            self.text_content += "\n\n" + content
        else:
            self.text_content = content
    
    def add_image(self, image_b64: str, description: str = ""):
        """Add image to the response"""
        self.images.append({
            "data": image_b64,
            "description": description,
            "format": "base64"
        })
    
    def add_audio(self, audio_data: str, description: str = "", audio_format: str = "audio"):
        """Add audio to the response"""
        self.audio.append({
            "data": audio_data,
            "description": description,
            "format": audio_format
        })
    
    def to_pollinations_format(self, request_format: str = "auto") -> Union[Dict[str, Any], bytes]:
        """Convert to Pollinations-compatible response format"""
        
        # If only images, return binary image or image metadata
        if self.images and not self.text_content.strip() and not self.audio:
            if request_format == "binary":
                # Return raw image bytes for GET endpoints
                return base64.b64decode(self.images[0]["data"])
            else:
                return {
                    "id": f"moe_{int(time.time())}",
                    "object": "image.generation",
                    "created": int(time.time()),
                    "model": "moe-unified",
                    "data": [{
                        "b64_json": self.images[0]["data"],
                        "url": None
                    }],
                    "moe_metadata": {
                        "models_used": self.models_used,
                        "strategy": self.strategy,
                        "total_latency": self.latency
                    }
                }
        
        # If only audio, return audio format  
        elif self.audio and not self.text_content.strip() and not self.images:
            return {
                "id": f"moe_{int(time.time())}",
                "object": "audio.generation",
                "created": int(time.time()),
                "model": "moe-unified",
                "data": self.audio[0]["data"],
                "format": self.audio[0]["format"],
                "moe_metadata": {
                    "models_used": self.models_used,
                    "strategy": self.strategy,
                    "total_latency": self.latency
                }
            }
        
        # If only text, return OpenAI chat format
        elif self.text_content and not self.images and not self.audio:
            return {
                "id": f"moe_{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "moe-unified",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": self.text_content,
                        "refusal": None
                    },
                    "logprobs": None,
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 100,  # Estimated
                    "completion_tokens": len(self.text_content.split()),
                    "total_tokens": 100 + len(self.text_content.split())
                },
                "moe_metadata": {
                    "models_used": self.models_used,
                    "strategy": self.strategy,
                    "total_latency": self.latency
                }
            }
        
        # Full multimodal response - text + images + audio
        else:
            multimodal_content = self.text_content if self.text_content else "Generated multimodal response:"
            
            # Add image references to text
            for i, img in enumerate(self.images):
                multimodal_content += f"\n\n[Generated Image {i+1}]"
                if img.get("description"):
                    multimodal_content += f" - {img['description']}"
                multimodal_content += f"\n![Image {i+1}](data:image/jpeg;base64,{img['data'][:50]}...)"
            
            # Add audio references
            for i, audio in enumerate(self.audio):
                multimodal_content += f"\n\n[Generated Audio {i+1}]"
                if audio.get("description"):
                    multimodal_content += f" - {audio['description']}"
                multimodal_content += f"\n🔊 Audio content available"
            
            return {
                "id": f"moe_{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "moe-unified",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": multimodal_content,
                        "refusal": None
                    },
                    "logprobs": None,
                    "finish_reason": "stop"
                }],
                "images": self.images if self.images else None,
                "audio": self.audio if self.audio else None,
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": len(multimodal_content.split()),
                    "total_tokens": 100 + len(multimodal_content.split())
                },
                "moe_metadata": {
                    "models_used": self.models_used,
                    "strategy": self.strategy,
                    "total_latency": self.latency,
                    "multimodal": True,
                    "modalities": {
                        "text": bool(self.text_content),
                        "images": len(self.images),
                        "audio": len(self.audio)
                    }
                }
            }

@app.on_event("startup")
async def startup_event():
    """Initialize the MoE model on startup"""
    logger.info("Starting MoE Unified Model...")
    logger.info(f"Loaded {len(registry.text_models)} text models and {len(registry.image_models)} image models")

@app.post("/openai")
async def moe_openai_endpoint(request: Request):
    """OpenAI-compatible endpoint - main entry point for MoE model"""
    try:
        request_data = await request.json()
        
        # Process with MoE orchestration
        async with client:
            result = await orchestrator.process_request(request_data)
            
            # Create unified response
            response = MultimodalResponse()
            response.models_used = result.models_used
            response.strategy = result.strategy.value if hasattr(result.strategy, 'value') else str(result.strategy)
            response.latency = result.total_latency
            
            # Extract content from orchestration result
            if isinstance(result.content, dict):
                # Handle multimodal workflow response
                if "text" in result.content and "images" in result.content:
                    # Full multimodal response
                    if result.content.get("text"):
                        response.add_text(result.content["text"])
                    
                    # Add images
                    for img in result.content.get("images", []):
                        response.add_image(img["data"], img.get("description", "Generated image"))
                    
                    # Add audio
                    for audio in result.content.get("audio", []):
                        response.add_audio(audio["data"], audio.get("description", "Generated audio"))
                        
                # Handle OpenAI-style response
                elif "choices" in result.content:
                    text_content = result.content["choices"][0]["message"]["content"]
                    response.add_text(text_content)
                    
                # Handle pure image response
                elif "image" in result.content:
                    response.add_image(result.content["image"], "Generated image")
                    
                # Handle image generation response format
                elif "data" in result.content and isinstance(result.content["data"], list):
                    for img_data in result.content["data"]:
                        if "b64_json" in img_data:
                            response.add_image(img_data["b64_json"], "Generated image")
                        
                # Handle audio response
                elif "audio" in result.content:
                    response.add_audio(result.content["audio"], "Generated audio")
                    
                else:
                    # Fallback for unknown formats
                    response.add_text(str(result.content))
            else:
                response.add_text(str(result.content))
            
            return response.to_pollinations_format()
            
    except Exception as e:
        logger.error(f"Error in MoE processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "moe-unified",
        "version": "1.0.0",
        "capabilities": {
            "text_generation": True,
            "image_generation": True,
            "multimodal": True,
            "context_aware": True
        },
        "models_available": len(registry.text_models) + len(registry.image_models)
    }

@app.get("/{prompt:path}")
async def moe_get_endpoint(prompt: str, request: Request):
    """GET endpoint for simple prompts - Pollinations style"""
    try:
        # Get query parameters
        query_params = dict(request.query_params)
        
        # Convert GET request to internal format
        internal_request = {
            "messages": [{"role": "user", "content": prompt}],
            "model": "auto",
            **query_params
        }
        
        # Process with MoE orchestration
        async with client:
            result = await orchestrator.process_request(internal_request)
            
            # For GET requests, try to return appropriate format
            if result.strategy.value == "image_generation" or query_params.get("format") == "image":
                # Return binary image if image generation
                if hasattr(result.content, 'get') and result.content.get("image"):
                    image_data = base64.b64decode(result.content["image"])
                    return Response(
                        content=image_data,
                        media_type="image/jpeg",
                        headers={
                            "X-MoE-Models": ",".join(result.models_used),
                            "X-MoE-Strategy": result.strategy.value,
                            "X-MoE-Latency": str(result.total_latency)
                        }
                    )
            
            # Return text response
            if isinstance(result.content, dict) and "choices" in result.content:
                text_content = result.content["choices"][0]["message"]["content"]
            else:
                text_content = str(result.content)
                
            return Response(
                content=text_content,
                media_type="text/plain",
                headers={
                    "X-MoE-Models": ",".join(result.models_used),
                    "X-MoE-Strategy": result.strategy.value,
                    "X-MoE-Latency": str(result.total_latency)
                }
            )
            
    except Exception as e:
        logger.error(f"Error in GET processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "moe_model_endpoint:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
