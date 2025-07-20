"""
Pollinations API Client
Handles all communication with Pollinations text and image APIs
"""

import asyncio
import aiohttp
import json
import base64
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging
from dotenv import load_dotenv

from models_registry import ModelInfo

# Load environment variables
load_dotenv()

@dataclass
class APIResponse:
    success: bool
    content: Any
    model: str
    latency: float
    error: Optional[str] = None
    raw_response: Optional[Dict] = None

class PollinationsClient:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # API endpoints - Updated to match actual Pollinations API
        self.text_base_url = "https://text.pollinations.ai"  
        self.image_base_url = "https://image.pollinations.ai"
        
        # Authentication
        self.token = os.getenv("TOKEN")
        self.referrer = os.getenv("REFERRER", "https://pollinations.ai")
        
        if not self.token:
            raise ValueError("TOKEN must be set in environment variables")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for requests"""
        return {
            "Content-Type": "application/json",
            "User-Agent": "MoE-API/1.0",
            "Referer": self.referrer
        }
    
    def _get_auth_params(self) -> Dict[str, str]:
        """Get authentication parameters"""
        params = {}
        if self.referrer:
            params["referrer"] = self.referrer
        return params
        
    async def __aenter__(self):
        # Create session with no timeout to handle long AI model responses
        timeout = aiohttp.ClientTimeout(total=None)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def chat_completion(self, model: str, messages: List[Dict], 
                            temperature: float = 0.7, max_tokens: Optional[int] = None,
                            top_p: Optional[float] = None, top_k: Optional[int] = None,
                            presence_penalty: Optional[float] = None, 
                            frequency_penalty: Optional[float] = None,
                            tools: Optional[List[Dict]] = None, **kwargs) -> APIResponse:
        """Make a chat completion request to Pollinations text API (POST)"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=None)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
        # Use the correct Pollinations POST endpoint
        url = f"{self.text_base_url}/openai"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }
        
        # Add all OpenAI-compatible parameters
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if tools:
            payload["tools"] = tools
        
        # Add voice parameter for openai-audio model
        if "voice" in kwargs:
            payload["voice"] = kwargs["voice"]
        
        # Add referrer for authentication
        auth_params = self._get_auth_params()
        payload.update(auth_params)
        
        try:
            start_time = asyncio.get_event_loop().time()
            headers = self._get_headers()
            
            async with self.session.post(url, json=payload, headers=headers) as response:
                latency = asyncio.get_event_loop().time() - start_time
                
                if response.status == 200:
                    response_data = await response.json()
                    return APIResponse(
                        success=True,
                        content=response_data,
                        model=model,
                        latency=latency,
                        raw_response=response_data
                    )
                else:
                    error_text = await response.text()
                    return APIResponse(
                        success=False,
                        content=None,
                        model=model,
                        latency=latency,
                        error=f"HTTP {response.status}: {error_text}",
                        raw_response={"error": error_text}
                    )
                    
        except Exception as e:
            return APIResponse(
                success=False,
                content=None,
                model=model,
                latency=0.0,
                error=str(e)
            )
    
    async def generate_image(self, model: str, prompt: str, 
                           image_input: Optional[str] = None,
                           width: int = 1024, height: int = 1024,
                           **kwargs) -> APIResponse:
        """Generate image using Pollinations image API (GET with URL-encoded prompt)"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=None)
            self.session = aiohttp.ClientSession(timeout=timeout)
        
        # URL encode the prompt as per Pollinations API docs
        import urllib.parse
        encoded_prompt = urllib.parse.quote(prompt)
        
        # Build URL with encoded prompt
        url = f"{self.image_base_url}/prompt/{encoded_prompt}"
        
        # Build query parameters
        params = {
            "model": model,
            "width": width,
            "height": height
        }
        
        # Add image input for img2img (should be a URL)
        if image_input:
            params["image"] = image_input
        
        # Add authentication via referrer    
        auth_params = self._get_auth_params()
        params.update(auth_params)
        params.update(kwargs)
        
        try:
            start_time = asyncio.get_event_loop().time()
            headers = self._get_headers()
            
            async with self.session.get(url, params=params, headers=headers) as response:
                latency = asyncio.get_event_loop().time() - start_time
                
                if response.status == 200:
                    # For images, we get the binary content
                    image_data = await response.read()
                    image_b64 = base64.b64encode(image_data).decode('utf-8')
                    
                    return APIResponse(
                        success=True,
                        content={
                            "image": image_b64,
                            "url": str(response.url),
                            "format": response.headers.get('content-type', 'image/jpeg')
                        },
                        model=model,
                        latency=latency
                    )
                else:
                    error_text = await response.text()
                    return APIResponse(
                        success=False,
                        content=None,
                        model=model,
                        latency=latency,
                        error=f"HTTP {response.status}: {error_text}"
                    )
                    
        except Exception as e:
            return APIResponse(
                success=False,
                content=None,
                model=model,
                latency=0.0,
                error=str(e)
            )
    
    def _add_moe_system_prompt(self, messages: List[Dict], is_multimodal: bool = False) -> List[Dict]:
        """Add default MoE system prompt to prevent AI models from denying capabilities"""
        
        if is_multimodal:
            moe_system_prompt = {
                "role": "system", 
                "content": """You are part of an advanced MoE (Mixture of Experts) system that can generate text, images, and audio. 
                
When users request multimodal content (text + images + audio):
- Focus on providing excellent text content for your part
- Do NOT say "I can't create images" or "I can't generate audio" 
- The MoE system will automatically handle image and audio generation
- If asked for images/audio, describe what should be created and the system will generate it
- Provide comprehensive text responses as your contribution to the multimodal output

You are the text expert in this collaborative AI system."""
            }
        else:
            moe_system_prompt = {
                "role": "system",
                "content": """You are part of an advanced MoE (Mixture of Experts) system. Provide the best possible response for your expertise area. Do not mention limitations about other modalities - focus on delivering excellent results in your domain."""
            }
        
        # Add MoE system prompt first, then user's system prompt if any
        enhanced_messages = [moe_system_prompt]
        
        # Add user's system prompt after the MoE prompt (if it exists)
        for msg in messages:
            if msg.get("role") == "system":
                enhanced_messages.append(msg)
            
        # Add all non-system messages
        enhanced_messages.extend([msg for msg in messages if msg.get("role") != "system"])
        
        return enhanced_messages

    async def make_request(self, model_info: ModelInfo, request_data: Dict[str, Any]) -> APIResponse:
        """Generic method to make requests based on model type"""
        
        if model_info.name in ["flux", "kontext", "turbo", "gptimage"]:
            # Image generation - no system prompt needed
            prompt = request_data.get("prompt", "")
            image_input = request_data.get("image")
            
            return await self.generate_image(
                model=model_info.name,
                prompt=prompt,
                image_input=image_input,
                width=request_data.get("width", 1024),
                height=request_data.get("height", 1024)
            )
        else:
            # Text/Audio generation - add MoE system prompt
            messages = request_data.get("messages", [])
            if not messages and request_data.get("prompt"):
                messages = [{"role": "user", "content": request_data["prompt"]}]
            
            # Detect if this is part of multimodal workflow
            is_multimodal = request_data.get("is_multimodal_workflow", False)
            
            # Add MoE system prompt
            enhanced_messages = self._add_moe_system_prompt(messages, is_multimodal)
            
            # Handle voice parameter for openai-audio model
            chat_params = {
                "model": model_info.name,
                "messages": enhanced_messages,
                "temperature": request_data.get("temperature", 0.7),
                "max_tokens": request_data.get("max_tokens"),
                "top_p": request_data.get("top_p"),
                "top_k": request_data.get("top_k"),
                "presence_penalty": request_data.get("presence_penalty"),
                "frequency_penalty": request_data.get("frequency_penalty"),
                "tools": request_data.get("tools")
            }
            
            # Add voice parameter for openai-audio model
            if model_info.name == "openai-audio" and request_data.get("voice"):
                chat_params["voice"] = request_data.get("voice")
            
            return await self.chat_completion(**chat_params)
    
    async def batch_requests(self, requests: List[tuple[ModelInfo, Dict[str, Any]]]) -> List[APIResponse]:
        """Make multiple requests concurrently"""
        tasks = [
            self.make_request(model_info, request_data) 
            for model_info, request_data in requests
        ]
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def format_openai_response(self, response: APIResponse, request_format: str = "openai") -> Dict[str, Any]:
        """Format response to match OpenAI API format"""
        if not response.success:
            return {
                "error": {
                    "message": response.error,
                    "type": "api_error",
                    "code": "model_error"
                }
            }
        
        if response.model in ["flux", "kontext", "turbo", "gptimage"]:
            # Image response - format as OpenAI image generation response
            return {
                "created": int(asyncio.get_event_loop().time()),
                "data": [
                    {
                        "b64_json": response.content["image"],
                        "url": response.content.get("url")
                    }
                ]
            }
        else:
            # Text response - already in OpenAI format from Pollinations
            return response.content

# Global client instance
client = PollinationsClient()
