"""
Pollinations AI Models Registry
Comprehensive registry of all available models with their capabilities
"""

import json
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from enum import Enum

class ModelTier(Enum):
    ANONYMOUS = "anonymous"
    SEED = "seed" 
    FLOWER = "flower"
    NECTAR = "nectar"

class Modality(Enum):
    TEXT = "text"
    IMAGE = "image" 
    AUDIO = "audio"

@dataclass
class ModelInfo:
    name: str
    description: str
    provider: str
    tier: ModelTier
    input_modalities: Set[Modality]
    output_modalities: Set[Modality]
    tools: bool = False
    vision: bool = False
    audio: bool = False
    reasoning: bool = False
    uncensored: bool = False
    community: bool = False
    aliases: Optional[str] = None
    max_input_chars: Optional[int] = None
    voices: Optional[List[str]] = None

class ModelsRegistry:
    def __init__(self):
        self.text_models: Dict[str, ModelInfo] = {}
        self.image_models: Dict[str, ModelInfo] = {}
        self.audio_models: Dict[str, ModelInfo] = {}
        self.load_models()
    
    def load_models(self):
        """Load and parse model data from JSON files"""
        # Load text models
        with open('text_models.json', 'r') as f:
            text_data = json.load(f)
        
        for model in text_data:
            model_info = ModelInfo(
                name=model['name'],
                description=model['description'],
                provider=model['provider'],
                tier=ModelTier(model['tier']),
                input_modalities=set(Modality(m) for m in model['input_modalities']),
                output_modalities=set(Modality(m) for m in model['output_modalities']),
                tools=model.get('tools', False),
                vision=model.get('vision', False),
                audio=model.get('audio', False),
                reasoning=model.get('reasoning', False),
                uncensored=model.get('uncensored', False),
                community=model.get('community', False),
                aliases=model.get('aliases'),
                max_input_chars=model.get('maxInputChars'),
                voices=model.get('voices')
            )
            self.text_models[model['name']] = model_info
            
        # Load image models with detailed capabilities and preferences
        image_models_data = {
            "flux": {
                "description": "Flux 1 Schnell - Fast image generation model (seconds)",
                "input_modalities": [Modality.TEXT],
                "img2img": False,
                "capabilities": ["fast", "general", "text2img", "quick_generation"],
                "preferred_for": ["quick_generation", "simple_images", "speed_priority"],
                "generation_time": "seconds",
                "complexity_level": "standard",
                "best_for": "Fast, simple to moderate complexity images"
            },
            "gptimage": {
                "description": "GPTImage Model 1 - Handles complex images expertly (3-5 minutes)",
                "input_modalities": [Modality.TEXT, Modality.IMAGE],
                "img2img": True,
                "aliases": "gptimage-model-1",
                "capabilities": ["complex_tasks", "text_understanding", "multi_image", "intelligent_prompts", "complex_images"],
                "preferred_for": ["complex_images", "detailed_work", "text_in_images", "multi_image_editing", "high_quality"],
                "multi_image_support": True,
                "generation_time": "3-5 minutes",
                "complexity_level": "high",
                "best_for": "Complex, detailed images with high quality requirements"
            },
            "turbo": {
                "description": "Turbo Model (likely SDXL Turbo) - Fast generation", 
                "input_modalities": [Modality.TEXT],
                "img2img": False,
                "capabilities": ["fast", "sdxl", "general"],
                "preferred_for": ["speed", "general_images"]
            },
            "kontext": {
                "description": "Flux 1 Kontext Dev - Preferred for image inpainting and editing (1 image at a time)",
                "input_modalities": [Modality.TEXT, Modality.IMAGE],
                "img2img": True,
                "aliases": "flux-kontext-dev",
                "capabilities": ["inpainting", "editing", "img2img", "single_image_only"],
                "preferred_for": ["inpainting", "image_editing", "single_image_modifications"],
                "multi_image_support": False,
                "constraint": "Only accepts 1 image at a time"
            }
        }
            
        for model_name, model_data in image_models_data.items():
            model_info = ModelInfo(
                name=model_name,
                description=model_data["description"],
                provider="pollinations",
                tier=ModelTier.ANONYMOUS,
                input_modalities=set(model_data["input_modalities"]),
                output_modalities={Modality.IMAGE},
                vision=model_data.get("img2img", False),
                aliases=model_data.get("aliases")
            )
            self.image_models[model_name] = model_info
    
    def get_models_by_capability(self, capability: str) -> List[ModelInfo]:
        """Get models that support a specific capability"""
        models = []
        for model in list(self.text_models.values()) + list(self.image_models.values()):
            if capability == "tools" and model.tools:
                models.append(model)
            elif capability == "vision" and model.vision:
                models.append(model)
            elif capability == "audio" and model.audio:
                models.append(model)
            elif capability == "reasoning" and model.reasoning:
                models.append(model)
            elif capability == "uncensored" and model.uncensored:
                models.append(model)
            elif capability == "img2img" and Modality.IMAGE in model.input_modalities and Modality.IMAGE in model.output_modalities:
                models.append(model)
        return models
    
    def get_models_by_modality(self, input_modality: Modality, output_modality: Modality) -> List[ModelInfo]:
        """Get models that support specific input/output modalities"""
        models = []
        for model in list(self.text_models.values()) + list(self.image_models.values()):
            if input_modality in model.input_modalities and output_modality in model.output_modalities:
                models.append(model)
        return models
    
    def get_best_models_for_task(self, task_type: str, modalities: Dict[str, str]) -> List[ModelInfo]:
        """Get the best models for a specific task type"""
        input_mod = Modality(modalities.get('input', 'text'))
        output_mod = Modality(modalities.get('output', 'text'))
        
        candidates = self.get_models_by_modality(input_mod, output_mod)
        
        # Sort by preference (non-community, higher tier, specific capabilities)
        def score_model(model: ModelInfo) -> int:
            score = 0
            if not model.community:
                score += 10
            if model.tier == ModelTier.ANONYMOUS:
                score += 5
            elif model.tier == ModelTier.SEED:
                score += 4
            elif model.tier == ModelTier.FLOWER:
                score += 3
            elif model.tier == ModelTier.NECTAR:
                score += 2
                
            # Task-specific scoring
            if task_type == "reasoning" and model.reasoning:
                score += 15
            if task_type == "creative" and model.uncensored:
                score += 10
            if task_type == "technical" and model.tools:
                score += 8
            if task_type == "img2img" and Modality.IMAGE in model.input_modalities:
                score += 12
                
            return score
        
        return sorted(candidates, key=score_model, reverse=True)
    
    def get_model_by_name(self, name: str) -> Optional[ModelInfo]:
        """Get model by name or alias"""
        # Check exact name match first
        if name in self.text_models:
            return self.text_models[name]
        if name in self.image_models:
            return self.image_models[name]
            
        # Check aliases
        for model in list(self.text_models.values()) + list(self.image_models.values()):
            if model.aliases and name == model.aliases:
                return model
                
        return None
    
    def get_summary(self) -> Dict:
        """Get a summary of all available models"""
        return {
            "total_text_models": len(self.text_models),
            "total_image_models": len(self.image_models),
            "models_with_tools": len(self.get_models_by_capability("tools")),
            "models_with_vision": len(self.get_models_by_capability("vision")),
            "models_with_audio": len(self.get_models_by_capability("audio")),
            "models_with_reasoning": len(self.get_models_by_capability("reasoning")),
            "uncensored_models": len(self.get_models_by_capability("uncensored")),
            "img2img_models": len(self.get_models_by_capability("img2img")),
            "providers": list(set(m.provider for m in list(self.text_models.values()) + list(self.image_models.values())))
        }

# Global registry instance
registry = ModelsRegistry()
