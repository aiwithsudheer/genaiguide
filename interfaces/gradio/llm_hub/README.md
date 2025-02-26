# LLM Hub - Multi-Model Chat Interface

A unified chat interface that allows interaction with multiple Large Language Models through a single, elegant interface.

## Features

- Support for multiple LLM providers:
  - OpenAI (GPT-4, GPT-3.5)
  - Anthropic (Claude)
  - Google (Gemini)
  - Mistral AI
- Real-time streaming responses
- System prompt configuration
- Temperature control
- Clean, modern UI
- Markdown support
- Copy functionality
- Chat history management

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables in `.env`:
```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
MISTRAL_API_KEY=your_mistral_key
```

3. Run the interface:
```bash
python llm_hub.py
```

## Usage

1. Select your preferred model from the dropdown
2. (Optional) Set a system message to guide the AI's behavior
3. Adjust the temperature slider for response creativity
4. Type your message and press Enter or click Send
5. View the streamed response in real-time

## Technical Details

- Built with Gradio and LangChain
- Supports real-time streaming
- Implements proper error handling
- Maintains conversation history
- Responsive design with custom CSS 