"""
MoE Orchestrator - Multi-model coordination and response combination
Handles different routing strategies and combines responses intelligently
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging
import re

from models_registry import registry, ModelInfo
from moe_router import router, RoutingDecision, RoutingStrategy, TaskType
from pollinations_client import client, APIResponse
from performance_tracker import performance_tracker

@dataclass
class OrchestrationResult:
    content: Any
    models_used: List[str]
    strategy: RoutingStrategy
    total_latency: float
    individual_responses: List[APIResponse]
    reasoning: str = ""

class MoEOrchestrator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = client
        self.router = router
        
    async def process_request(self, request: Dict[str, Any]) -> OrchestrationResult:
        """Main orchestration method - routes and processes requests"""
        
        # Step 1: Get routing decision
        routing_decision = await self.router.route_request(request)
        
        # Step 2: Execute based on strategy
        start_time = asyncio.get_event_loop().time()
        
        if routing_decision.strategy == RoutingStrategy.SINGLE_BEST:
            result = await self._execute_single_best(request, routing_decision)
        elif routing_decision.strategy == RoutingStrategy.MULTI_MODEL:
            result = await self._execute_multi_model(request, routing_decision)
        elif routing_decision.strategy == RoutingStrategy.SPECIALIZED_CHAIN:
            result = await self._execute_specialized_chain(request, routing_decision)
        elif routing_decision.strategy == RoutingStrategy.PARALLEL_VOTING:
            result = await self._execute_parallel_voting(request, routing_decision)
        elif routing_decision.strategy == RoutingStrategy.MULTIMODAL_WORKFLOW:
            result = await self._execute_multimodal_workflow(request, routing_decision)
        elif routing_decision.strategy == RoutingStrategy.ENHANCED_RESPONSE:
            result = await self._execute_enhanced_response(request, routing_decision)
        else:
            result = await self._execute_single_best(request, routing_decision)
        
        total_latency = asyncio.get_event_loop().time() - start_time
        
        # Log performance for intelligent optimization
        self._log_orchestration_performance(
            routing_decision, 
            result["models_used"], 
            total_latency,
            len(str(result["content"]))
        )
        
        return OrchestrationResult(
            content=result["content"],
            models_used=result["models_used"],
            strategy=routing_decision.strategy,
            total_latency=total_latency,
            individual_responses=result["responses"],
            reasoning=routing_decision.reasoning
        )
    
    async def _execute_single_best(self, request: Dict[str, Any], 
                                 routing_decision: RoutingDecision) -> Dict[str, Any]:
        """Execute single best model strategy with optimized parameters"""
        
        model = routing_decision.primary_models[0]
        
        # Apply MoE parameter optimization
        optimized_request = self._optimize_request_parameters(request, routing_decision)
        
        response = await self.client.make_request(model, optimized_request)
        
        return {
            "content": response.content,
            "models_used": [model.name],
            "responses": [response]
        }
    
    async def _execute_multi_model(self, request: Dict[str, Any], 
                                 routing_decision: RoutingDecision) -> Dict[str, Any]:
        """Execute multi-model strategy - combine responses from multiple models"""
        
        models = routing_decision.primary_models[:3]  # Limit to 3 models
        
        # Apply MoE parameter optimization
        optimized_request = self._optimize_request_parameters(request, routing_decision)
        
        # Make requests to all models concurrently
        requests = [(model, optimized_request) for model in models]
        responses = await self.client.batch_requests(requests)
        
        # Filter successful responses
        successful_responses = [r for r in responses if isinstance(r, APIResponse) and r.success]
        
        if not successful_responses:
            # Fallback to single best if all failed
            return await self._execute_single_best(request, routing_decision)
        
        # Combine responses based on task type
        combined_content = await self._combine_responses(
            successful_responses, 
            routing_decision.task_type, 
            request
        )
        
        return {
            "content": combined_content,
            "models_used": [r.model for r in successful_responses],
            "responses": successful_responses
        }
    
    async def _execute_specialized_chain(self, request: Dict[str, Any], 
                                       routing_decision: RoutingDecision) -> Dict[str, Any]:
        """Execute specialized chain - chain models for complex tasks"""
        
        primary_model = routing_decision.primary_models[0]
        responses = []
        
        # Step 1: Primary model processes the request
        primary_response = await self.client.make_request(primary_model, request)
        responses.append(primary_response)
        
        if not primary_response.success or not routing_decision.secondary_models:
            return {
                "content": primary_response.content,
                "models_used": [primary_model.name],
                "responses": responses
            }
        
        # Step 2: Secondary model refines/enhances the response
        secondary_model = routing_decision.secondary_models[0]
        
        # Create refinement request
        if routing_decision.task_type == TaskType.CODE_GENERATION:
            refinement_prompt = f"""
            Review and improve this code response:
            
            {self._extract_text_content(primary_response.content)}
            
            Original request: {self._extract_prompt(request)}
            
            Please provide an enhanced version with:
            - Better error handling
            - Improved documentation
            - Optimization suggestions
            """
        else:
            refinement_prompt = f"""
            Enhance and refine this response:
            
            {self._extract_text_content(primary_response.content)}
            
            Original request: {self._extract_prompt(request)}
            
            Please provide an improved version that is more comprehensive and accurate.
            """
        
        refinement_request = {
            "messages": [{"role": "user", "content": refinement_prompt}]
        }
        
        secondary_response = await self.client.make_request(secondary_model, refinement_request)
        responses.append(secondary_response)
        
        # Use secondary response if successful, otherwise primary
        final_content = secondary_response.content if secondary_response.success else primary_response.content
        
        return {
            "content": final_content,
            "models_used": [primary_model.name, secondary_model.name],
            "responses": responses
        }
    
    async def _execute_parallel_voting(self, request: Dict[str, Any], 
                                     routing_decision: RoutingDecision) -> Dict[str, Any]:
        """Execute parallel voting - run multiple models and combine best aspects"""
        
        models = routing_decision.primary_models[:3]  # Max 3 models for voting
        
        # Make requests to all models concurrently
        requests = [(model, request) for model in models]
        responses = await self.client.batch_requests(requests)
        
        # Filter successful responses
        successful_responses = [r for r in responses if isinstance(r, APIResponse) and r.success]
        
        if not successful_responses:
            return await self._execute_single_best(request, routing_decision)
        
        # Vote and combine responses
        combined_content = await self._vote_and_combine(
            successful_responses, 
            routing_decision.task_type,
            request
        )
        
        return {
            "content": combined_content,
            "models_used": [r.model for r in successful_responses],
            "responses": successful_responses
        }
    
    async def _combine_responses(self, responses: List[APIResponse], 
                               task_type: TaskType, request: Dict[str, Any]) -> Any:
        """Combine multiple responses intelligently based on task type"""
        
        if task_type == TaskType.IMAGE_GENERATION:
            # For images, return the best one (could implement voting later)
            return responses[0].content
        
        if task_type == TaskType.CODE_GENERATION:
            # Combine code responses
            return await self._combine_code_responses(responses, request)
        
        if task_type == TaskType.CREATIVE_WRITING:
            # Merge creative content
            return await self._combine_creative_responses(responses, request)
        
        # Default: combine text responses
        return await self._combine_text_responses(responses, request)
    
    async def _combine_text_responses(self, responses: List[APIResponse], 
                                    request: Dict[str, Any]) -> Dict[str, Any]:
        """Combine multiple text responses"""
        
        # Extract text content from each response
        texts = []
        for response in responses:
            content = self._extract_text_content(response.content)
            if content:
                texts.append(f"**{response.model}**: {content}")
        
        # Create combined response
        combined_text = "\n\n".join(texts)
        
        # Use the format of the first successful response
        base_response = responses[0].content.copy()
        if "choices" in base_response:
            base_response["choices"][0]["message"]["content"] = combined_text
        else:
            base_response = {"content": combined_text}
        
        return base_response
    
    async def _combine_code_responses(self, responses: List[APIResponse], 
                                    request: Dict[str, Any]) -> Dict[str, Any]:
        """Combine code responses intelligently"""
        
        # Extract code blocks from responses
        code_blocks = []
        explanations = []
        
        for response in responses:
            content = self._extract_text_content(response.content)
            
            # Extract code blocks
            code_pattern = r'```(?:\w+)?\n(.*?)```'
            codes = re.findall(code_pattern, content, re.DOTALL)
            
            if codes:
                code_blocks.extend(codes)
            
            # Extract explanation (text outside code blocks)
            explanation = re.sub(code_pattern, '', content, flags=re.DOTALL).strip()
            if explanation:
                explanations.append(f"**{response.model}**: {explanation}")
        
        # Combine the best code (longest/most complete) with all explanations
        best_code = max(code_blocks, key=len) if code_blocks else ""
        combined_explanation = "\n\n".join(explanations)
        
        combined_content = f"{combined_explanation}\n\n```python\n{best_code}\n```"
        
        # Format response
        base_response = responses[0].content.copy()
        if "choices" in base_response:
            base_response["choices"][0]["message"]["content"] = combined_content
        else:
            base_response = {"content": combined_content}
        
        return base_response
    
    async def _combine_creative_responses(self, responses: List[APIResponse], 
                                        request: Dict[str, Any]) -> Dict[str, Any]:
        """Combine creative writing responses"""
        
        # For creative content, we can either:
        # 1. Return the longest/most detailed response
        # 2. Blend elements from multiple responses
        
        contents = [self._extract_text_content(r.content) for r in responses]
        contents = [c for c in contents if c]  # Filter empty
        
        if not contents:
            return responses[0].content
        
        # Return the most detailed response (longest)
        best_content = max(contents, key=len)
        
        # Format response
        base_response = responses[0].content.copy()
        if "choices" in base_response:
            base_response["choices"][0]["message"]["content"] = best_content
        else:
            base_response = {"content": best_content}
        
        return base_response
    
    async def _vote_and_combine(self, responses: List[APIResponse], 
                              task_type: TaskType, request: Dict[str, Any]) -> Any:
        """Vote on responses and combine the best aspects"""
        
        # For now, implement a simple scoring system
        scored_responses = []
        
        for response in responses:
            content = self._extract_text_content(response.content)
            score = 0
            
            # Length bonus (more comprehensive)
            score += len(content) * 0.001
            
            # Model-specific bonuses
            if response.model in ["openai-fast", "openai", "deepseek"]:
                score += 10
            elif response.model in ["qwen-coder"] and task_type == TaskType.CODE_GENERATION:
                score += 15
            elif response.model in ["grok", "mistral"] and task_type == TaskType.REASONING:
                score += 12
            
            # Performance bonus (lower latency)
            score += max(0, 10 - response.latency)
            
            scored_responses.append((score, response))
        
        # Sort by score and return the best response
        scored_responses.sort(key=lambda x: x[0], reverse=True)
        best_response = scored_responses[0][1]
        
        return best_response.content
    
    async def _execute_multimodal_workflow(self, request: Dict[str, Any], 
                                         routing_decision: RoutingDecision) -> Dict[str, Any]:
        """Execute multimodal workflow - generate text, images, and/or audio as needed"""
        
        responses = []
        models_used = []
        multimodal_content = {
            "text": "",
            "images": [],
            "audio": []
        }
        
        # Step 1: Generate comprehensive text response
        text_models = [m for m in routing_decision.primary_models if m.name not in ["flux", "kontext", "turbo", "gptimage"]][:2]
        if text_models:
            text_request = request.copy()
            # Mark as multimodal workflow for system prompt
            text_request["is_multimodal_workflow"] = True
            text_response = await self.client.make_request(text_models[0], text_request)
            if text_response.success:
                responses.append(text_response)
                models_used.append(text_response.model)
                multimodal_content["text"] = self._extract_text_content(text_response.content)
        
        # Step 2: Generate relevant image if needed
        image_prompt = self._extract_image_prompt_from_request(request)
        if image_prompt and self._should_generate_image(request):
            # Use intelligent image model selection from router
            from moe_router import MoERouter
            router = MoERouter()
            
            # Create a simplified classification for image model selection
            image_classification = {
                "enhanced_params": routing_decision.enhanced_params if hasattr(routing_decision, 'enhanced_params') else {},
                "reasoning": image_prompt
            }
            
            image_models = router._select_optimal_image_model(image_classification)
            image_model = image_models[0] if image_models else registry.get_model_by_name("flux")
            
            if image_model:
                image_request = {"prompt": image_prompt, "width": 1024, "height": 1024}
                image_response = await self.client.make_request(image_model, image_request)
                if image_response.success:
                    responses.append(image_response)
                    models_used.append(image_response.model)
                    multimodal_content["images"].append({
                        "data": image_response.content.get("image", ""),
                        "description": f"Generated illustration: {image_prompt}",
                        "format": "base64"
                    })
        
        # Step 3: Generate audio if requested (Text-to-Speech)
        should_generate_audio = self._should_generate_audio(request)
        self.logger.info(f"🔊 Audio check: should_generate={should_generate_audio}, text_length={len(multimodal_content.get('text', ''))}")
        
        if should_generate_audio:
            if multimodal_content["text"]:
                self.logger.info(f"🔊 Generating audio narration for {len(multimodal_content['text'])} chars of text")
                # Generate audio narration of the text content
                audio_response = await self._generate_audio_narration(
                    multimodal_content["text"], 
                    voice="nova"
                )
                if audio_response and audio_response.success:
                    self.logger.info("🔊 Audio narration generated successfully!")
                    responses.append(audio_response)
                    models_used.append("openai-audio")
                    multimodal_content["audio"].append({
                        "data": audio_response.content.get("audio", ""),
                        "description": "Generated audio narration",
                        "format": "audio/mpeg",
                        "voice": "nova"
                    })
                else:
                    self.logger.warning(f"🔊 Audio generation failed: {audio_response.error if audio_response else 'No response'}")
            else:
                self.logger.warning("🔊 No text content available for audio generation")
        
        return {
            "content": multimodal_content,
            "models_used": models_used,
            "responses": responses
        }
    
    async def _execute_enhanced_response(self, request: Dict[str, Any], 
                                       routing_decision: RoutingDecision) -> Dict[str, Any]:
        """Execute enhanced response - multiple text models for comprehensive answers"""
        
        models = routing_decision.primary_models[:3]
        text_models = [m for m in models if m.name not in ["flux", "kontext", "turbo", "gptimage"]]
        
        if not text_models:
            return await self._execute_single_best(request, routing_decision)
        
        # Make requests to multiple text models
        requests = [(model, request) for model in text_models]
        responses = await self.client.batch_requests(requests)
        
        # Filter successful responses
        successful_responses = [r for r in responses if isinstance(r, APIResponse) and r.success]
        
        if not successful_responses:
            return await self._execute_single_best(request, routing_decision)
        
        # Create enhanced combined response
        enhanced_content = await self._create_enhanced_response(successful_responses, request)
        
        return {
            "content": enhanced_content,
            "models_used": [r.model for r in successful_responses],
            "responses": successful_responses
        }
    
    def _should_generate_image(self, request: Dict[str, Any]) -> bool:
        """Determine if an image should be generated for this request"""
        prompt = self._extract_prompt(request).lower()
        
        image_indicators = [
            'show', 'demonstrate', 'visualize', 'diagram', 'chart', 'graph',
            'illustration', 'picture', 'image', 'draw', 'sketch', 'design',
            'tutorial', 'guide', 'example', 'comparison', 'workflow'
        ]
        
        return any(indicator in prompt for indicator in image_indicators)
    
    def _should_generate_audio(self, request: Dict[str, Any]) -> bool:
        """Determine if audio should be generated for this request"""
        prompt = self._extract_prompt(request).lower()
        
        # Enhanced audio detection matching the router logic
        audio_indicators = [
            'read', 'speak', 'say', 'voice', 'audio', 'listen', 'hear',
            'pronunciation', 'narrate', 'tell me', 'read aloud', 'explain aloud',
            'read this', 'speak this', 'narration', 'voice over'
        ]
        
        # Check for audio phrases
        audio_phrases = ['read aloud', 'read this aloud', 'explain aloud', 'narrate this', 'voice over']
        
        # Basic keyword check
        has_audio_words = any(indicator in prompt for indicator in audio_indicators)
        
        # Phrase check for more specific requests
        has_audio_phrases = any(phrase in prompt for phrase in audio_phrases)
        
        return has_audio_words or has_audio_phrases
    
    def _extract_text_content(self, response_content: Any) -> str:
        """Extract text content from various response formats"""
        if isinstance(response_content, str):
            return response_content
        
        if isinstance(response_content, dict):
            # OpenAI format
            if "choices" in response_content and response_content["choices"]:
                choice = response_content["choices"][0]
                if "message" in choice:
                    return choice["message"].get("content", "")
                elif "text" in choice:
                    return choice["text"]
            
            # Direct content
            if "content" in response_content:
                return response_content["content"]
        
        return str(response_content)
    
    def _extract_prompt(self, request: Dict[str, Any]) -> str:
        """Extract prompt from request"""
        if "prompt" in request:
            return request["prompt"]
        
        if "messages" in request and request["messages"]:
            return request["messages"][-1].get("content", "")
        
        return ""
    
    def _extract_image_prompt_from_request(self, request: Dict[str, Any]) -> str:
        """Extract or generate an appropriate image prompt from the request"""
        prompt = self._extract_prompt(request)
        
        # Visual content indicators
        visual_indicators = {
            'diagram': 'technical diagram illustrating',
            'chart': 'informative chart showing',
            'tutorial': 'step-by-step visual guide for',
            'example': 'clear example illustration of',
            'demonstrate': 'visual demonstration of',
            'show me': 'detailed visual representation of',
            'explain': 'explanatory diagram about',
            'code': 'programming flowchart for',
            'algorithm': 'algorithm visualization of',
            'process': 'process flow diagram of',
            'comparison': 'side-by-side comparison chart of'
        }
        
        for indicator, prefix in visual_indicators.items():
            if indicator in prompt.lower():
                subject = prompt.lower().replace(indicator, '').strip()[:100]
                return f"{prefix} {subject}"
        
        # Default contextual image
        return f"Professional illustration related to: {prompt[:100]}"
    
    def _extract_audio_prompt_from_request(self, request: Dict[str, Any], text_content: str) -> str:
        """Extract or generate appropriate audio content from request"""
        prompt = self._extract_prompt(request)
        
        audio_keywords = ['read', 'speak', 'say', 'voice', 'audio', 'pronunciation', 'listen']
        
        if any(keyword in prompt.lower() for keyword in audio_keywords):
            if text_content:
                return f"Please read this content aloud: {text_content[:500]}"
            else:
                return prompt
        
        # Default: narrate the text response
        if text_content:
            return f"Provide an audio summary of: {text_content[:300]}"
        
        return prompt
    
    async def _create_enhanced_response(self, responses: List[APIResponse], 
                                      request: Dict[str, Any]) -> Dict[str, Any]:
        """Create enhanced response by intelligently combining multiple outputs"""
        
        # Extract all text responses
        texts = []
        for response in responses:
            content = self._extract_text_content(response.content)
            if content:
                texts.append(content)
        
        if not texts:
            return responses[0].content
        
        # Take the most comprehensive response as the base
        primary_text = max(texts, key=len)
        
        # Add insights from other models if they provide unique value
        enhanced_text = primary_text
        for text in texts[1:]:
            if len(text) > len(primary_text) * 0.3:  # Only add substantial additions
                unique_parts = self._extract_unique_insights(text, primary_text)
                if unique_parts:
                    enhanced_text += f"\n\n**Additional Insights:** {unique_parts}"
        
        # Format as standard response
        base_response = responses[0].content.copy() if responses[0].content else {}
        if "choices" in base_response:
            base_response["choices"][0]["message"]["content"] = enhanced_text
        else:
            base_response = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": enhanced_text
                    }
                }]
            }
        
        return base_response
    
    def _extract_unique_insights(self, text: str, primary_text: str) -> str:
        """Extract unique insights from secondary text that aren't in primary"""
        # Simple approach: look for sentences not in primary
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        unique_sentences = []
        
        for sentence in sentences:
            if len(sentence) > 20 and sentence.lower() not in primary_text.lower():
                unique_sentences.append(sentence)
        
        return '. '.join(unique_sentences[:2])  # Limit to 2 unique insights

    def _optimize_request_parameters(self, request: Dict[str, Any], 
                                   routing_decision: RoutingDecision) -> Dict[str, Any]:
        """Dynamically optimize request parameters based on task analysis - MoE/GoE intelligence"""
        
        optimized_request = request.copy()
        
        # Get enhanced parameters from AI analysis
        enhanced_params = getattr(routing_decision, 'enhanced_params', {})
        if not enhanced_params:
            # Fallback to basic optimization
            enhanced_params = {
                "temperature": 0.7,
                "creativity_level": "medium",
                "complexity_level": "moderate"
            }
        
        # Dynamic parameter optimization based on task analysis
        task_type = routing_decision.task_type
        creativity = enhanced_params.get("creativity_level", "medium")
        complexity = enhanced_params.get("complexity_level", "moderate")
        
        # Optimize temperature for task type and creativity
        if task_type == TaskType.CODE_GENERATION:
            # Lower temperature for precise code
            optimized_request["temperature"] = 0.2
            optimized_request["top_p"] = 0.9
        elif task_type == TaskType.CREATIVE_WRITING:
            # Higher temperature for creativity
            optimized_request["temperature"] = 1.0 + (0.3 if creativity == "high" else 0.0)
            optimized_request["top_p"] = 0.95
            optimized_request["presence_penalty"] = 0.6  # Encourage diverse vocabulary
        elif task_type == TaskType.REASONING:
            # Balanced for logical thinking
            optimized_request["temperature"] = 0.3
            optimized_request["top_p"] = 0.85
            optimized_request["frequency_penalty"] = 0.1  # Reduce repetition
        elif task_type == TaskType.IMAGE_GENERATION:
            # Not applicable for image models, but keep for consistency
            pass
        else:  # TEXT_GENERATION
            # Dynamic based on complexity and creativity
            base_temp = 0.7
            
            # Adjust for creativity
            creativity_mult = {"low": 0.8, "medium": 1.0, "high": 1.3}.get(creativity, 1.0)
            
            # Adjust for complexity (lower temp for complex tasks)
            complexity_mult = {"simple": 1.1, "moderate": 1.0, "complex": 0.7}.get(complexity, 1.0)
            
            optimized_request["temperature"] = min(1.5, base_temp * creativity_mult * complexity_mult)
            optimized_request["top_p"] = 0.9
        
        # Set max_tokens based on complexity and response style
        response_style = enhanced_params.get("response_style", "detailed")
        token_mapping = {
            "concise": 500,
            "detailed": 1200, 
            "comprehensive": 2000
        }
        
        if not optimized_request.get("max_tokens"):
            optimized_request["max_tokens"] = token_mapping.get(response_style, 1200)
        
        # Apply presence/frequency penalties based on task
        if task_type in [TaskType.CREATIVE_WRITING, TaskType.TEXT_GENERATION]:
            if not optimized_request.get("presence_penalty"):
                # Encourage diverse vocabulary for creative tasks
                creativity_penalty = {"low": 0.0, "medium": 0.3, "high": 0.6}.get(creativity, 0.3)
                optimized_request["presence_penalty"] = creativity_penalty
                
            if not optimized_request.get("frequency_penalty"):
                # Reduce repetition especially for longer responses
                style_penalty = {"concise": 0.1, "detailed": 0.2, "comprehensive": 0.4}.get(response_style, 0.2)
                optimized_request["frequency_penalty"] = style_penalty
        
        # Optimize top_k for certain models (if supported)
        if task_type == TaskType.CODE_GENERATION:
            optimized_request["top_k"] = 10  # More focused for precise code
        elif creativity == "high":
            optimized_request["top_k"] = 100  # More diverse options
        
        temp = optimized_request.get('temperature', 'N/A')
        top_p = optimized_request.get('top_p', 'N/A') 
        max_tokens = optimized_request.get('max_tokens', 'N/A')
        
        temp_str = f"{temp:.2f}" if isinstance(temp, (int, float)) else str(temp)
        self.logger.info(f"🎛️ Optimized parameters: temp={temp_str}, "
                        f"top_p={top_p}, max_tokens={max_tokens}")
        
        return optimized_request
    
    async def _generate_audio_narration(self, text_content: str, voice: str = "nova") -> Optional[APIResponse]:
        """Generate audio narration using openai-audio model with POST (only images use GET)"""
        
        try:
            # Clean and truncate text for audio generation
            clean_text = text_content.strip()
            if len(clean_text) > 1000:  # Can handle longer text with POST
                clean_text = clean_text[:1000] + "..."
            
            # Create proper request for audio output using POST
            audio_request = {
                "model": "openai-audio",
                "messages": [
                    {
                        "role": "user", 
                        "content": clean_text  # Just the text content, no meta-instructions
                    }
                ],
                "voice": voice,
                "response_format": "audio",  # Specify we want audio output
                "temperature": 0.3  # Lower temp for consistent narration
            }
            
            self.logger.info(f"🔊 Generating audio with POST for {len(clean_text)} chars")
            
            # Get openai-audio model
            audio_model = registry.get_model_by_name("openai-audio")
            if not audio_model:
                self.logger.warning("🔊 openai-audio model not found")
                return None
            
            # Make request using POST endpoint (not GET)
            audio_response = await self.client.make_request(audio_model, audio_request)
            
            if audio_response.success:
                self.logger.info(f"🔊 Audio generated successfully with {voice} voice using POST")
                return audio_response
            else:
                self.logger.warning(f"🔊 Audio POST request failed: {audio_response.error}")
                return None
                
        except Exception as e:
            self.logger.error(f"🔊 Audio generation error: {e}")
            return None
    
    def _log_orchestration_performance(self, routing_decision: RoutingDecision, 
                                     models_used: List[str], total_latency: float, 
                                     content_length: int):
        """Log performance data for intelligent optimization"""
        
        # Get enhanced parameters for complexity tracking
        enhanced_params = {}
        if hasattr(routing_decision, 'enhanced_params'):
            enhanced_params = routing_decision.enhanced_params
        
        complexity = enhanced_params.get('complexity_level', 'moderate')
        
        # Log each model's contribution
        for model_name in models_used:
            performance_tracker.log_performance(
                model_name=model_name,
                task_type=routing_decision.task_type.value,
                strategy=routing_decision.strategy.value,
                response_time=total_latency,
                success=True,
                content_length=content_length,
                complexity_level=complexity,
                parameters=enhanced_params
            )
        
        # Log strategy performance
        performance_tracker.log_performance(
            model_name=f"strategy_{routing_decision.strategy.value}",
            task_type=routing_decision.task_type.value,
            strategy=routing_decision.strategy.value,
            response_time=total_latency,
            success=True,
            content_length=content_length,
            complexity_level=complexity,
            parameters={
                "models_count": len(models_used),
                "confidence": routing_decision.confidence,
                **enhanced_params
            }
        )

# Global orchestrator instance
orchestrator = MoEOrchestrator()
