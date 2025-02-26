# Gradio Interfaces

This directory contains various Gradio-based user interfaces and examples demonstrating different capabilities of the Gradio framework.

## Contents

### LLM Hub
- Location: `/llm_hub`
- A unified chat interface for multiple LLM providers
- Supports GPT-4, Claude, Gemini, and more

### Gradio Playground
- Location: `/gradio_playground.ipynb`
- Interactive notebook demonstrating Gradio features:
  - Basic interfaces
  - Image processing
  - Tabular data handling
  - Custom styling
  - Authentication
  - State management

## About Gradio

Gradio is an open-source Python library that makes it easy to create customizable UI components for machine learning models, data processing pipelines, and other Python functions. It allows you to:

- Create web interfaces with just a few lines of code
- Support for 30+ types of components (text, images, audio, video, etc.)
- Real-time interaction and streaming
- Easy sharing and deployment
- Integration with popular ML frameworks

### Documentation References

- [Quickstart Guide](https://www.gradio.app/guides/quickstart)
- [Building Interfaces](https://www.gradio.app/guides/creating-a-chatbot-fast)
- [Custom Components](https://www.gradio.app/guides/custom-components-in-five-minutes)
- [Streaming](https://www.gradio.app/guides/real-time-speech-recognition)
- [Blocks & Events](https://www.gradio.app/guides/blocks-and-event-listeners)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run any example:
```bash
python llm_hub/llm_hub.py
# or
jupyter notebook gradio_playground.ipynb
``` 