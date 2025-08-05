# MoE (Mixture of Experts) AI System

A sophisticated AI orchestration system that acts as a single unified endpoint while intelligently coordinating multiple AI models for optimal results. Provides context-aware processing similar to ChatGPT, Gemini, and Grok with full multimodal capabilities.

## 🚀 Features

### 🧠 Context-Aware Intelligence
- **AI-Powered Classification**: Uses advanced AI models to understand user intent and complexity
- **Dynamic Task Analysis**: Automatically assesses complexity, creativity needs, and optimal workflow
- **Intelligent Parameter Optimization**: Real-time adjustment of temperature, top_p, penalties, etc.

### 🤖 Multi-Model Orchestration  
- **26 Text Models + 4 Image Models**: Comprehensive model registry with detailed capabilities
- **Multiple Routing Strategies**: Single best, multi-model, specialized chains, parallel voting
- **Performance-Driven Selection**: Intelligent model selection based on task requirements

### 🌐 Full Multimodal Capabilities
- **Text Generation**: Advanced reasoning, coding, creative writing
- **Image Generation**: Flux (seconds) for simple images, GPTImage (3-5 min) for complex tasks
- **Audio Generation**: Text-to-speech narration with voice selection
- **Combined Workflows**: Text + Images + Audio in single responses

### ⚡ Production Ready
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI endpoints
- **Performance Tracking**: Intelligent logging system (10 logs max) for optimization
- **Error Handling**: Comprehensive fallback systems and error recovery
- **No Timeouts**: Handles long AI model responses without limits

## 🏗️ Architecture

<img width="3600" height="1000" alt="image" src="https://github.com/user-attachments/assets/bd1c8b97-58a2-404f-b723-3ce651840b64" />



## 🛠️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/Itachi-1824/MoE.git
cd MoE
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create/edit `.env` file:
```env
# Pollinations Authentication (optional but recommended)
POLLINATIONS_TOKEN=your_token_here
REFERRER=https://pollinations.ai
```

### 4. Start the MoE System
```bash
python start_moe_model.py
```

The system will start on `http://localhost:8000`

## 📡 API Usage

### OpenAI-Compatible Endpoint
```bash
curl -X POST http://localhost:8000/openai \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [
      {"role": "user", "content": "Explain quantum physics with a diagram and read it aloud"}
    ]
  }'
```

### Simple GET Endpoint
```bash
curl "http://localhost:8000/Create%20a%20diagram%20of%20the%20solar%20system"
```

### Available Parameters
- `model`: "auto" for intelligent selection or specific model name
- `messages`: OpenAI-compatible message format
- `temperature`: 0.0-2.0 (automatically optimized)
- `top_p`: 0.0-1.0 nucleus sampling
- `max_tokens`: Response length limit
- `presence_penalty`: -2.0 to 2.0
- `frequency_penalty`: -2.0 to 2.0

## 🎯 Intelligent Model Selection

### Text Models (26 available)
- **Simple Tasks**: openai-fast, llama-fast-roblox
- **Complex Reasoning**: deepseek-reasoning, grok, qwen-coder
- **Creative Tasks**: mistral, openai-large
- **Audio Output**: openai-audio (with voice selection)

### Image Models (4 available)
- **Flux**: Fast generation (seconds) for simple images
- **GPTImage**: Complex tasks (3-5 min) with text understanding
- **Kontext**: Image editing and inpainting (1 image limit)
- **Turbo**: SDXL-based fast generation

### Selection Logic
```
Simple request → Single fast model
Complex reasoning → Multiple premium models  
Creative task → High creativity models
Multimodal → Text + Image/Audio orchestration
```

## 📊 Response Format

```json
{
  "id": "chatcmpl-xyz",
  "object": "chat.completion", 
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Text response content"
    }
  }],
  "images": [{
    "data": "base64_encoded_image",
    "description": "Generated illustration", 
    "format": "base64"
  }],
  "audio": [{
    "data": "base64_encoded_audio",
    "format": "audio/mpeg",
    "voice": "nova"
  }],
  "moe_metadata": {
    "models_used": ["openai-fast", "gptimage", "openai-audio"],
    "strategy": "multimodal_workflow",
    "total_latency": 23.5,
    "modalities": {"text": true, "images": 1, "audio": 1}
  }
}
```

## 🔧 Core Components

### 1. **MoE Router** (`moe_router.py`)
- AI-powered prompt classification
- Task complexity assessment
- Model selection algorithms
- Parameter optimization

### 2. **MoE Orchestrator** (`moe_orchestrator.py`) 
- Multi-model coordination
- Response combination strategies
- Multimodal workflow management
- Performance logging

### 3. **Pollinations Client** (`pollinations_client.py`)
- API communication with Pollinations.ai
- Authentication handling
- POST for text/audio, GET for images
- Error handling and retries

### 4. **Models Registry** (`models_registry.py`)
- Comprehensive model database
- Capability-based selection
- Tier and performance metadata

### 5. **Performance Tracker** (`performance_tracker.py`)
- Response time optimization
- Success rate monitoring  
- Storage-efficient logging (10 log limit)

## 🎨 Multimodal Examples

### Text + Image Generation
```json
{
  "model": "auto",
  "messages": [{"role": "user", "content": "Show me how photosynthesis works"}]
}
```
→ Text explanation + scientific diagram

### Text + Image + Audio
```json
{
  "model": "auto", 
  "messages": [{"role": "user", "content": "Read this aloud and show a visual: How do rockets work?"}]
}
```
→ Text explanation + rocket diagram + audio narration

### Complex Reasoning
```json
{
  "model": "auto",
  "messages": [{"role": "user", "content": "Analyze the implications of quantum computing on cryptography"}]
}
```
→ Multi-model analysis (9,000+ chars vs 3,000 single model)

## 🚀 Performance Features

- **Intelligent Caching**: 10-log performance tracking for optimization
- **Dynamic Model Selection**: Fastest models recommended for urgent tasks
- **Real-time Optimization**: Parameter adjustment based on task analysis
- **Fallback Systems**: Multiple redundancy levels for reliability

## 📈 Production Deployment

The system is designed for production use with:
- Horizontal scaling capability
- Comprehensive error handling
- Performance monitoring
- OpenAI API compatibility
- No timeout limitations for long responses

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test thoroughly
4. Submit pull request with detailed description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 [Itachi-1824](https://github.com/Itachi-1824)

## 🔗 Related

- [Pollinations.ai API Docs](https://github.com/pollinations/pollinations/blob/master/APIDOCS.md)
- [OpenAI API Compatibility](https://platform.openai.com/docs/api-reference)

---

**Built with ❤️ using advanced AI orchestration techniques**
