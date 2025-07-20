"""
MoE Router - Intelligent model selection and orchestration
Uses OpenAI-fast for task classification and routing decisions
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging

from models_registry import registry, ModelInfo, Modality

class TaskType(Enum):
    TEXT_GENERATION = "text_generation"
    IMAGE_GENERATION = "image_generation"
    IMAGE_TO_IMAGE = "img2img"
    AUDIO_GENERATION = "audio_generation" 
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"
    CREATIVE_WRITING = "creative_writing"
    TECHNICAL_ANALYSIS = "technical_analysis"
    MULTIMODAL = "multimodal"
    MULTIMODAL_TEXT_IMAGE = "multimodal_text_image"  # New: text + image in one response
    IMAGE_EDIT = "image_edit"  # New: image editing/manipulation
    TOOL_USE = "tool_use"

class RoutingStrategy(Enum):
    SINGLE_BEST = "single_best"          # Use single best model
    MULTI_MODEL = "multi_model"          # Use multiple models and combine
    SPECIALIZED_CHAIN = "specialized_chain"  # Chain specialized models
    PARALLEL_VOTING = "parallel_voting"   # Run multiple models and vote
    MULTIMODAL_WORKFLOW = "multimodal_workflow"  # Text + Image generation
    ENHANCED_RESPONSE = "enhanced_response"  # Multiple text models + optional images

@dataclass
class RoutingDecision:
    task_type: TaskType
    strategy: RoutingStrategy
    primary_models: List[ModelInfo]
    secondary_models: List[ModelInfo] = None
    reasoning: str = ""
    confidence: float = 0.0

class MoERouter:
    def __init__(self):
        self.registry = registry
        self.logger = logging.getLogger(__name__)
        
        # Task classification prompts
        self.active_classification_prompt = """
        You are an intelligent AI workflow analyzer. Analyze the user request and determine the optimal AI workflow strategy.
        
        Your job is to understand:
        1. What the user actually wants (not just keywords)
        2. What combination of AI capabilities would best serve them
        3. How to orchestrate multiple models for the best result
        
        Available capabilities:
        - text_generation: Conversations, explanations, writing
        - image_generation: Creating visual content, diagrams, art
        - img2img: Editing or transforming existing images  
        - audio_generation: Voice, music, sound effects
        - reasoning: Complex analysis, math, logical thinking
        - code_generation: Programming, debugging, technical tasks
        - creative_writing: Stories, poems, creative content
        - multimodal_text_image: Text explanation WITH visual aid
        - multimodal_full: Text + images + audio together
        
        Analyze the context and intent, then return JSON:
        {
            "primary_task": "main task type",
            "secondary_tasks": ["additional helpful tasks"],
            "confidence": 0.0-1.0,
            "reasoning": "why this workflow makes sense",
            "suggested_models": ["specific model names that would work best"],
            "output_modalities": ["text", "image", "audio"],
            "parameters": {
                "temperature": 0.1-2.0,
                "creativity_level": "low|medium|high",
                "complexity": "simple|moderate|complex",
                "response_style": "concise|detailed|comprehensive"
            },
            "workflow_strategy": "single|parallel|chain|multimodal",
            "estimated_tokens": 50-2000,
            "should_enhance": true/false
        }
        
        Think like ChatGPT, Gemini, or Grok - understand user intent and create the optimal experience.
        """
    
    async def classify_task(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Actively classify the incoming request using AI models"""
        try:
            # Extract relevant information from request
            prompt = request.get('prompt', '') or request.get('messages', [{}])[-1].get('content', '')
            has_images = bool(request.get('image') or any(
                isinstance(msg.get('content'), list) and 
                any(item.get('type') == 'image_url' for item in msg.get('content', []))
                for msg in request.get('messages', [])
            ))
            
            self.logger.info(f"🧠 Classifying prompt: '{prompt[:50]}...' with images: {has_images}")
            
            # Try AI-powered classification first (disabled for debugging)
            # classification_result = await self._ai_classify_prompt(prompt, has_images)
            # For now, use the intelligent fallback which should work better
            classification_result = self._intelligent_fallback_classification(request)
            
            self.logger.info(f"✅ Classification result: {classification_result['task_type']} - {classification_result['reasoning']}")
            return classification_result
            
        except Exception as e:
            self.logger.error(f"❌ Classification failed: {e}")
            return self._emergency_fallback(request)
    
    async def _ai_classify_prompt(self, prompt: str, has_images: bool) -> Dict[str, Any]:
        """Use AI to actively analyze and classify the prompt"""
        
        # Import here to avoid circular imports
        from pollinations_client import client
        
        classification_request = {
            "model": "openai-fast",
            "messages": [
                {"role": "system", "content": self.active_classification_prompt},
                {"role": "user", "content": f"""
                Analyze this user request and determine the optimal AI workflow:
                
                Request: "{prompt}"
                Has images attached: {has_images}
                
                Consider:
                - What would provide the most value to the user?
                - Should this be just text, or would visuals/audio help?
                - What's the user's likely intent and context?
                - How can we exceed their expectations?
                
                Return the JSON analysis.
                """}
            ],
            "temperature": 0.3,  # Lower temperature for consistent analysis
            "response_format": {"type": "json_object"}
        }
        
        try:
            async with client:
                model_info = registry.get_model_by_name("openai-fast")
                if model_info:
                    response = await client.make_request(model_info, classification_request)
                    
                    if response.success:
                        # Extract the analysis from the AI response
                        ai_content = self._extract_ai_response_content(response.content)
                        try:
                            import json
                            analysis = json.loads(ai_content)
                            
                            # Convert AI analysis to our internal format
                            return self._convert_ai_analysis_to_internal(analysis, prompt, has_images)
                        except json.JSONDecodeError:
                            self.logger.warning("AI returned invalid JSON, using fallback")
                            return self._intelligent_fallback_classification({"messages": [{"content": prompt}]})
            
            # Fallback if AI classification fails
            return self._intelligent_fallback_classification({"messages": [{"content": prompt}]})
            
        except Exception as e:
            self.logger.error(f"AI classification error: {e}")
            return self._intelligent_fallback_classification({"messages": [{"content": prompt}]})
    
    def _extract_ai_response_content(self, response_content: Any) -> str:
        """Extract text content from AI response"""
        if isinstance(response_content, dict) and "choices" in response_content:
            return response_content["choices"][0]["message"]["content"]
        return str(response_content)
    
    def _convert_ai_analysis_to_internal(self, analysis: Dict[str, Any], prompt: str, has_images: bool) -> Dict[str, Any]:
        """Convert AI analysis to our internal classification format"""
        
        # Map AI analysis to our task types
        task_mapping = {
            "text_generation": "text_generation",
            "image_generation": "image_generation", 
            "img2img": "img2img",
            "audio_generation": "audio_generation",
            "reasoning": "reasoning",
            "code_generation": "code_generation",
            "creative_writing": "creative_writing",
            "multimodal_text_image": "multimodal_text_image",
            "multimodal_full": "multimodal_text_image"
        }
        
        primary_task = analysis.get("primary_task", "text_generation")
        mapped_task = task_mapping.get(primary_task, "text_generation")
        
        # Extract parameters with defaults
        params = analysis.get("parameters", {})
        temperature = params.get("temperature", 0.7)
        creativity = params.get("creativity_level", "medium")
        complexity = params.get("complexity", "moderate")
        style = params.get("response_style", "detailed")
        
        # Determine if multiple models should be used
        workflow = analysis.get("workflow_strategy", "single")
        use_multiple = workflow in ["parallel", "chain", "multimodal"]
        
        # Enhanced parameters for real-time optimization
        enhanced_params = {
            "temperature": temperature,
            "max_tokens": analysis.get("estimated_tokens", 1000),
            "creativity_level": creativity,
            "complexity_level": complexity,
            "response_style": style,
            "suggested_models": analysis.get("suggested_models", []),
            "workflow_strategy": workflow
        }
        
        return {
            "task_type": mapped_task,
            "confidence": analysis.get("confidence", 0.8),
            "reasoning": analysis.get("reasoning", "AI-analyzed request"),
            "input_modalities": ["image", "text"] if has_images else ["text"],
            "output_modalities": analysis.get("output_modalities", ["text"]),
            "complexity": complexity,
            "use_multiple_models": use_multiple or analysis.get("should_enhance", False),
            "ai_analysis": analysis,  # Store full AI analysis
            "enhanced_params": enhanced_params  # Store enhanced parameters
        }
    
    def _intelligent_fallback_classification(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligent fallback classification with basic analysis"""
        prompt = str(request.get('prompt', '') or request.get('messages', [{}])[-1].get('content', ''))
        has_images = bool(request.get('image'))
        
        # Basic intelligent analysis without AI model
        return self._analyze_prompt_context(prompt, has_images)
    
    def _analyze_prompt_context(self, prompt: str, has_images: bool) -> Dict[str, Any]:
        """Analyze prompt context intelligently"""
        prompt_lower = prompt.lower()
        
        # Analyze intent and complexity
        complexity = self._assess_complexity(prompt)
        creativity_needed = self._assess_creativity_need(prompt)
        visual_benefit = self._assess_visual_benefit(prompt)
        audio_benefit = self._assess_audio_benefit(prompt)
        
        # Image generation keywords
        image_keywords = ['generate image', 'create image', 'create a', 'draw', 'picture of', 'image of', 'visualize', 'illustration', 'diagram', 'chart', 'make an image', 'design', 'sketch']
        img2img_keywords = ['modify image', 'edit image', 'transform', 'style transfer', 'change image', 'enhance image', 'fix image']
        reasoning_keywords = ['solve', 'calculate', 'analyze', 'reasoning', 'logic', 'math', 'problem', 'explain step by step']
        code_keywords = ['code', 'program', 'function', 'debug', 'python', 'javascript', 'algorithm', 'implement', 'development']
        creative_keywords = ['story', 'poem', 'creative', 'write', 'fiction', 'novel', 'narrative', 'character']
        tool_keywords = ['search', 'lookup', 'find', 'get data', 'api', 'function']
        
        # Multimodal keywords - requests that benefit from both text AND images
        multimodal_keywords = ['explain with', 'show me', 'demonstrate', 'tutorial', 'guide', 'how to', 'example', 'comparison']
        enhanced_response_keywords = ['detailed', 'comprehensive', 'thorough', 'complete', 'full explanation', 'in-depth']
        
        # Advanced classification with multimodal detection
        if has_images and any(kw in prompt for kw in img2img_keywords):
            task_type = "img2img"
            input_modalities = ["text", "image"]
            output_modalities = ["image"]
        elif any(kw in prompt for kw in image_keywords):
            # Check if this should be multimodal (text + image)
            if any(kw in prompt for kw in multimodal_keywords):
                task_type = "multimodal_text_image"
                input_modalities = ["text"]
                output_modalities = ["text", "image"]
            else:
                task_type = "image_generation"
                input_modalities = ["text"]
                output_modalities = ["image"]
        elif any(kw in prompt for kw in reasoning_keywords):
            # Complex reasoning might benefit from diagrams/visualizations
            if any(kw in prompt for kw in ['diagram', 'chart', 'visual', 'graph', 'show']):
                task_type = "multimodal_text_image"
                input_modalities = ["text"]
                output_modalities = ["text", "image"]
            else:
                task_type = "reasoning"
                input_modalities = ["text"]
                output_modalities = ["text"]
        elif any(kw in prompt for kw in code_keywords):
            # Code tutorials might benefit from visual examples
            if any(kw in prompt for kw in multimodal_keywords + ['tutorial', 'example', 'demonstrate']):
                task_type = "multimodal_text_image"
                input_modalities = ["text"]
                output_modalities = ["text", "image"]
            else:
                task_type = "code_generation"
                input_modalities = ["text"]
                output_modalities = ["text"]
        elif any(kw in prompt for kw in creative_keywords):
            task_type = "creative_writing"
            input_modalities = ["text"]
            output_modalities = ["text"]
        elif any(kw in prompt for kw in tool_keywords):
            task_type = "tool_use"
            input_modalities = ["text"]
            output_modalities = ["text"]
        else:
            # Check if this is asking for enhanced/comprehensive response
            if any(kw in prompt for kw in enhanced_response_keywords) and len(prompt.split()) > 50:
                task_type = "multimodal"
                input_modalities = ["text"]
                output_modalities = ["text"]
            else:
                task_type = "text_generation"
                input_modalities = ["text"]
                output_modalities = ["text"]
        
        # Determine optimal workflow based on analysis
        if has_images and any(kw in prompt_lower for kw in img2img_keywords):
            task_type = "img2img"
            output_modalities = ["image"]
        elif any(kw in prompt_lower for kw in image_keywords) or visual_benefit > 0.6:
            # Check if user wants explanation WITH image (multimodal)
            explanation_keywords = ['explain', 'describe', 'tell me', 'what is', 'how does', 'show me']
            wants_explanation = any(kw in prompt_lower for kw in explanation_keywords)
            
            if wants_explanation or visual_benefit > 0.6:  # Lower threshold for multimodal
                task_type = "multimodal_text_image"
                output_modalities = ["text", "image"]
            else:
                task_type = "image_generation"  
                output_modalities = ["image"]
        elif any(kw in prompt_lower for kw in reasoning_keywords) or complexity == "complex":
            task_type = "reasoning"
            output_modalities = ["text"]
        elif any(kw in prompt_lower for kw in code_keywords):
            task_type = "code_generation"
            output_modalities = ["text"]
        elif creativity_needed > 0.7:
            task_type = "creative_writing"
            output_modalities = ["text"]
        elif audio_benefit > 0.3:  # Lower threshold for audio detection
            if visual_benefit > 0.3:  # Both audio and visual 
                task_type = "multimodal_text_image" 
                output_modalities = ["text", "image", "audio"]
            else:
                task_type = "multimodal_text_image"  # Audio-focused multimodal
                output_modalities = ["text", "audio"]
        else:
            task_type = "text_generation"
            output_modalities = ["text"]
        
        # Dynamic parameters based on analysis
        temperature = self._calculate_optimal_temperature(creativity_needed, complexity)
        use_multiple = self._should_use_multiple_models(complexity, creativity_needed, visual_benefit)
        
        enhanced_params = {
            "temperature": temperature,
            "max_tokens": min(2000, max(200, len(prompt.split()) * 20)),
            "creativity_level": "high" if creativity_needed > 0.7 else "medium" if creativity_needed > 0.4 else "low",
            "complexity_level": complexity,
            "response_style": "comprehensive" if len(prompt.split()) > 30 else "detailed",
            "workflow_strategy": "multimodal" if visual_benefit > 0.6 else "chain" if complexity == "complex" else "single"
        }
        
        return {
            "task_type": task_type,
            "confidence": 0.7,  # Lower confidence for fallback
            "reasoning": f"Analyzed as {task_type} with {complexity} complexity and {creativity_needed:.1f} creativity need",
            "input_modalities": ["image", "text"] if has_images else ["text"],
            "output_modalities": output_modalities,
            "complexity": complexity,
            "use_multiple_models": use_multiple,
            "enhanced_params": enhanced_params
        }
    
    def _assess_complexity(self, prompt: str) -> str:
        """Assess the complexity of the request"""
        factors = 0
        prompt_lower = prompt.lower()
        
        # Length factor
        word_count = len(prompt.split())
        if word_count > 50: factors += 1
        if word_count > 100: factors += 1
        
        # Complexity indicators
        complex_words = ['analyze', 'explain', 'compare', 'evaluate', 'detailed', 'comprehensive', 
                        'step by step', 'in depth', 'thoroughly', 'algorithm', 'implement',
                        'implications', 'analysis', 'solutions', 'provide analysis']
        factors += sum(1 for word in complex_words if word in prompt_lower)
        
        # Technical terms
        if any(term in prompt_lower for term in ['code', 'programming', 'algorithm', 'technical', 'engineering']):
            factors += 1
            
        # Mathematical/logical terms  
        if any(term in prompt_lower for term in ['math', 'calculate', 'solve', 'logic', 'proof']):
            factors += 1
        
        return "complex" if factors >= 4 else "moderate" if factors >= 2 else "simple"
    
    def _assess_creativity_need(self, prompt: str) -> float:
        """Assess how much creativity is needed (0.0 to 1.0)"""
        creativity_score = 0.0
        prompt_lower = prompt.lower()
        
        # Creative indicators
        creative_words = ['create', 'creative', 'imagine', 'story', 'poem', 'art', 'design', 
                         'innovative', 'original', 'unique', 'brainstorm', 'ideas']
        creativity_score += sum(0.2 for word in creative_words if word in prompt_lower)
        
        # Factual indicators (reduce creativity)
        factual_words = ['fact', 'data', 'information', 'definition', 'what is', 'explain']
        creativity_score -= sum(0.1 for word in factual_words if word in prompt_lower)
        
        return max(0.0, min(1.0, creativity_score))
    
    def _assess_visual_benefit(self, prompt: str) -> float:
        """Assess if visual content would benefit this request (0.0 to 1.0)"""
        visual_score = 0.0
        prompt_lower = prompt.lower()
        
        # Strong visual indicators
        strong_visual = ['diagram', 'chart', 'graph', 'image', 'picture', 'visual', 'show me', 
                        'illustration', 'flowchart', 'timeline', 'map', 'show me a visual',
                        'create a diagram', 'draw', 'sketch', 'visualize']
        visual_score += sum(0.3 for word in strong_visual if word in prompt_lower)
        
        # Specific visual phrases
        visual_phrases = ['show me a visual', 'show me an image', 'create a diagram', 
                         'draw a picture', 'make a chart']
        visual_score += sum(0.4 for phrase in visual_phrases if phrase in prompt_lower)
        
        # Moderate visual indicators
        moderate_visual = ['example', 'demonstrate', 'tutorial', 'guide', 'how to', 'process', 
                          'workflow', 'comparison', 'structure']
        visual_score += sum(0.2 for word in moderate_visual if word in prompt_lower)
        
        # Educational/explanatory content often benefits from visuals
        if any(word in prompt_lower for word in ['explain', 'teach', 'learn', 'understand']):
            visual_score += 0.2
            
        return min(1.0, visual_score)
    
    def _assess_audio_benefit(self, prompt: str) -> float:
        """Assess if audio content would benefit this request (0.0 to 1.0)"""
        audio_score = 0.0
        prompt_lower = prompt.lower()
        
        # Audio indicators  
        audio_words = ['read', 'speak', 'say', 'voice', 'audio', 'listen', 'hear', 
                      'pronunciation', 'narrate', 'tell me', 'read aloud', 'explain aloud',
                      'read this', 'speak this', 'narration', 'voice over']
        audio_score += sum(0.3 for word in audio_words if word in prompt_lower)
        
        # Specific audio phrases
        audio_phrases = ['read aloud', 'read this aloud', 'explain aloud', 'narrate this', 'voice over']
        audio_score += sum(0.5 for phrase in audio_phrases if phrase in prompt_lower)
        
        return min(1.0, audio_score)
    
    def _calculate_optimal_temperature(self, creativity_needed: float, complexity: str) -> float:
        """Calculate optimal temperature based on creativity and complexity needs"""
        base_temp = 0.7
        
        # Adjust for creativity
        creativity_adjustment = (creativity_needed - 0.5) * 0.6  # -0.3 to +0.3 range
        
        # Adjust for complexity (lower temp for complex tasks)
        complexity_adjustment = {"simple": 0.1, "moderate": 0.0, "complex": -0.2}[complexity]
        
        return max(0.1, min(1.5, base_temp + creativity_adjustment + complexity_adjustment))
    
    def _should_use_multiple_models(self, complexity: str, creativity_needed: float, visual_benefit: float) -> bool:
        """Determine if multiple models should be used - MoE intelligence"""
        factors = 0
        
        # Complex tasks always benefit from multiple models
        if complexity == "complex": factors += 3
        if complexity == "moderate": factors += 2  # Increased from 1
        
        # Creative tasks benefit from multiple perspectives
        if creativity_needed > 0.7: factors += 2
        if creativity_needed > 0.4: factors += 1  # Medium creativity also benefits
        
        # Visual tasks often need multimodal approach
        if visual_benefit > 0.6: factors += 2
        if visual_benefit > 0.3: factors += 1
        
        # Lower threshold for using multiple models (was 2, now 1)
        return factors >= 1
    
    def _select_optimal_image_model(self, classification: Dict[str, Any]) -> List[ModelInfo]:
        """Intelligently select image model based on complexity and requirements"""
        
        # Extract task characteristics
        enhanced_params = classification.get("enhanced_params", {})
        complexity = enhanced_params.get("complexity_level", "moderate")
        creativity = enhanced_params.get("creativity_level", "medium")
        
        # Get original prompt for analysis
        prompt = classification.get("reasoning", "")  # Use reasoning field as fallback
        
        # Complex image indicators
        complex_indicators = [
            "detailed", "complex", "intricate", "realistic", "photorealistic",
            "high quality", "professional", "artistic", "fine art", "masterpiece",
            "text in image", "multiple objects", "scene with", "composition",
            "lighting", "shadows", "perspective", "3d render", "detailed background",
            "highly detailed", "cyberpunk", "futuristic", "multiple characters",
            "text signs", "architecture", "cityscape", "landscape", "portrait"
        ]
        
        # Simple/fast image indicators  
        simple_indicators = [
            "simple", "quick", "basic", "sketch", "outline", "logo",
            "icon", "minimal", "clean", "abstract", "geometric", "cartoon"
        ]
        
        prompt_lower = prompt.lower()
        complex_score = sum(1 for indicator in complex_indicators if indicator in prompt_lower)
        simple_score = sum(1 for indicator in simple_indicators if indicator in prompt_lower)
        
        # Decision logic (more aggressive for complex tasks)
        use_gptimage = False
        
        # Factors that favor gptimage (3-5 min, handles complex images expertly)
        if complex_score >= 1:  # Even one complexity indicator should trigger GPTImage
            use_gptimage = True
            reason = f"Complex image indicators detected ({complex_score})"
        elif complexity == "complex":  # High-level complexity assessment
            use_gptimage = True
            reason = "Complex task classification"
        elif creativity == "high":  # High creativity often needs quality
            use_gptimage = True
            reason = "High creativity requirement"
        elif "text" in prompt_lower and ("in" in prompt_lower or "on" in prompt_lower):  # Text in images
            use_gptimage = True
            reason = "Text-in-image requirement"
        elif len(prompt.split()) > 15:  # Long, detailed prompts (lowered threshold)
            use_gptimage = True
            reason = "Detailed prompt requires GPTImage intelligence"
        
        # Factors that favor flux (seconds, fast but limited complexity)
        elif simple_score >= 1:  # Simple image requests
            use_gptimage = False
            reason = "Simple image request - Flux speed advantage"
        elif complexity == "simple":
            use_gptimage = False
            reason = "Simple complexity level"
        elif "quick" in prompt_lower or "fast" in prompt_lower:
            use_gptimage = False
            reason = "Speed priority requested"
        
        # Default decision based on complexity
        elif complexity == "complex":
            use_gptimage = True
            reason = "Complex task - GPTImage preferred"
        else:
            use_gptimage = False  # Default to flux for speed
            reason = "Default to Flux for speed"
        
        # Get the selected model
        if use_gptimage:
            gptimage = self.registry.get_model_by_name("gptimage")
            self.logger.info(f"🎨 Selected GPTImage (3-5min): {reason}")
            return [gptimage] if gptimage else []
        else:
            flux = self.registry.get_model_by_name("flux")  
            self.logger.info(f"🚀 Selected Flux (seconds): {reason}")
            return [flux] if flux else []
    
    def _emergency_fallback(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Emergency fallback classification"""
        return {
            "task_type": "text_generation",
            "confidence": 0.5,
            "reasoning": "Emergency fallback - classification system failed",
            "input_modalities": ["text"],
            "output_modalities": ["text"],
            "complexity": "moderate",
            "use_multiple_models": True,  # Use multiple models for safety
            "enhanced_params": {
                "temperature": 0.7,
                "max_tokens": 1000,
                "creativity_level": "medium",
                "complexity_level": "moderate",
                "response_style": "detailed",
                "workflow_strategy": "multi_model"
            }
        }
    
    async def route_request(self, request: Dict[str, Any]) -> RoutingDecision:
        """Main routing logic - decides which models to use and how"""
        
        # Step 1: Classify the task
        classification = await self.classify_task(request)
        task_type = TaskType(classification["task_type"])
        
        # Step 2: Determine routing strategy
        strategy = self._determine_strategy(classification)
        
        # Step 3: Select models based on task type and strategy
        primary_models = self._select_primary_models(task_type, classification)
        secondary_models = self._select_secondary_models(task_type, classification, strategy)
        
        return RoutingDecision(
            task_type=task_type,
            strategy=strategy,
            primary_models=primary_models,
            secondary_models=secondary_models or [],
            reasoning=classification["reasoning"],
            confidence=classification["confidence"]
        )
    
    def _determine_strategy(self, classification: Dict[str, Any]) -> RoutingStrategy:
        """Determine the best routing strategy for the task"""
        
        task_type = classification["task_type"]
        
        # Multimodal tasks need special workflow
        if task_type == "multimodal_text_image":
            return RoutingStrategy.MULTIMODAL_WORKFLOW
        elif task_type == "multimodal" and classification.get("use_multiple_models", False):
            return RoutingStrategy.ENHANCED_RESPONSE
        elif classification["use_multiple_models"]:
            if classification["complexity"] == "high":
                return RoutingStrategy.SPECIALIZED_CHAIN
            else:
                return RoutingStrategy.MULTI_MODEL
        elif task_type in ["reasoning", "technical_analysis"]:
            return RoutingStrategy.PARALLEL_VOTING
        else:
            return RoutingStrategy.SINGLE_BEST
    
    def _select_primary_models(self, task_type: TaskType, classification: Dict[str, Any]) -> List[ModelInfo]:
        """Select primary models for the task"""
        
        input_mod = Modality(classification["input_modalities"][0])
        output_mod = Modality(classification["output_modalities"][0])
        
        if task_type == TaskType.IMAGE_GENERATION:
            return self._select_optimal_image_model(classification)
        
        elif task_type == TaskType.IMAGE_TO_IMAGE:
            return self.registry.get_models_by_capability("img2img")
        
        elif task_type == TaskType.REASONING:
            return self.registry.get_best_models_for_task("reasoning", {
                "input": classification["input_modalities"][0],
                "output": classification["output_modalities"][0]
            })[:3]
        
        elif task_type == TaskType.CODE_GENERATION:
            candidates = self.registry.get_best_models_for_task("technical", {
                "input": "text", "output": "text"
            })
            # Prefer qwen-coder for coding tasks
            qwen_coder = self.registry.get_model_by_name("qwen-coder")
            if qwen_coder and qwen_coder not in candidates[:3]:
                candidates.insert(0, qwen_coder)
            return candidates[:2]
        
        elif task_type == TaskType.CREATIVE_WRITING:
            return self.registry.get_best_models_for_task("creative", {
                "input": "text", "output": "text"
            })[:2]
        
        elif task_type == TaskType.TOOL_USE:
            return self.registry.get_models_by_capability("tools")[:3]
        
        else:  # TEXT_GENERATION and others
            # Get enhanced parameters to guide model selection
            enhanced_params = classification.get("enhanced_params", {})
            complexity = enhanced_params.get("complexity_level", "moderate")
            
            # Select models based on complexity and quality
            if complexity == "complex" or classification.get("use_multiple_models", False):
                # Use top-tier models for complex tasks
                preferred_models = ["openai-fast", "openai", "deepseek-reasoning", "qwen-coder", "mistral"]
                selected_models = []
                
                for model_name in preferred_models:
                    model = self.registry.get_model_by_name(model_name)
                    if model:
                        selected_models.append(model)
                        if len(selected_models) >= 2:
                            break
                
                if selected_models:
                    return selected_models
            else:
                # Single model for simple tasks - but use quality models
                preferred_single = ["openai-fast", "openai", "deepseek", "mistral", "qwen-coder"]
                for model_name in preferred_single:
                    model = self.registry.get_model_by_name(model_name)
                    if model:
                        return [model]
            
            # Final fallback to any available models
            return list(self.registry.text_models.values())[:1]
    
    def _select_secondary_models(self, task_type: TaskType, classification: Dict[str, Any], 
                               strategy: RoutingStrategy) -> Optional[List[ModelInfo]]:
        """Select secondary models for multi-model strategies"""
        
        if strategy == RoutingStrategy.SINGLE_BEST:
            return None
        
        # For complex tasks, add complementary models
        if task_type == TaskType.REASONING:
            # Add a creative model for explanation
            creative_models = self.registry.get_best_models_for_task("creative", {
                "input": "text", "output": "text"
            })
            return creative_models[:1]
        
        elif task_type == TaskType.CODE_GENERATION:
            # Add a reasoning model for logic checking
            reasoning_models = self.registry.get_models_by_capability("reasoning")
            return reasoning_models[:1]
        
        return None
    
    def get_model_endpoints(self, model: ModelInfo) -> Dict[str, str]:
        """Get API endpoints for a model"""
        if model.name in self.registry.text_models:
            return {
                "base_url": "https://text.pollinations.ai",
                "endpoint": f"/v1/chat/completions",
                "model_param": model.name
            }
        elif model.name in self.registry.image_models:
            return {
                "base_url": "https://image.pollinations.ai", 
                "endpoint": f"/prompt/{model.name}",
                "model_param": model.name
            }
        else:
            raise ValueError(f"Unknown model: {model.name}")

# Global router instance
router = MoERouter()
