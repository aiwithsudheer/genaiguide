# AI Framework Examples

A comprehensive collection of practical implementations and examples using various AI frameworks and tools. This repository demonstrates real-world applications of AI technologies through working examples and detailed documentation.

## Contents

### OpenFGA Integration (RAG with Access Control)
- **Location**: `/frameworks/open-fga`
- **Description**: Implements secure document question-answering with fine-grained access control
- **Key Features**:
  - RAG (Retrieval Augmented Generation) implementation
  - Document-level access control using OpenFGA
  - Secure document processing and embedding
  - Permission-aware document retrieval
  - Integration with Qdrant vector store
  - Comprehensive error handling
- [Detailed Documentation](frameworks/open-fga/README.md)

### Gradio Interfaces
- **Location**: `/interfaces/gradio`
- **Description**: Collection of user interfaces built with Gradio framework
- **Components**:
  1. **LLM Hub** (`/llm_hub`):
     - Multi-model chat interface
     - Supports GPT-4, Claude, Gemini, Mistral
     - Real-time streaming responses
     - System prompt configuration
     - Temperature control
     - Modern, responsive UI
  2. **Gradio Playground** (`/gradio_playground.ipynb`):
     - Interactive examples
     - Basic to advanced interfaces
     - Image processing demos
     - Tabular data handling
     - Custom styling examples
     - Authentication demos
- [Detailed Documentation](interfaces/gradio/README.md)

## Technologies Used

### Frameworks & Libraries
- **LangChain**: For RAG implementation and LLM integration
- **OpenFGA**: Fine-grained authorization
- **Gradio**: UI component creation
- **Qdrant**: Vector storage
- **LangGraph**: Conversation flow management

### Language Models
- OpenAI (GPT-4, GPT-3.5)
- Anthropic Claude
- Google Gemini
- Mistral AI

### Development Tools
- Python 3.9+
- Jupyter Notebooks
- Environment management
- Comprehensive logging

## Getting Started

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Git
- Access to API keys for various LLM providers

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-framework-examples.git
cd ai-framework-examples
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```

4. Configure your `.env` file with necessary API keys:
```env
# OpenAI
OPENAI_API_KEY=your_openai_key

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_key

# Google
GOOGLE_API_KEY=your_google_key

# Mistral
MISTRAL_API_KEY=your_mistral_key

# Qdrant
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key

# OpenFGA
OPENFGA_API_URL=your_openfga_url
OPENFGA_STORE_ID=your_store_id
```

## Directory Structure

```
.
├── frameworks/
│   └── open-fga/              # OpenFGA RAG implementation
│       ├── build_rag_with_fga.py
│       ├── README.md
│       └── requirements.txt
├── interfaces/
│   └── gradio/               # Gradio UI implementations
│       ├── llm_hub/         # Multi-model chat interface
│       │   ├── llm_hub.py
│       │   └── README.md
│       ├── gradio_playground.ipynb
│       └── README.md
├── requirements.txt
├── .env.example
└── README.md
```

## Usage Examples

### Running the LLM Hub
```bash
cd interfaces/gradio/llm_hub
python llm_hub.py
```

### Using RAG with OpenFGA
```bash
cd frameworks/open-fga
python build_rag_with_fga.py
```

### Exploring Gradio Examples
```bash
cd interfaces/gradio
jupyter notebook gradio_playground.ipynb
```

## Development Guidelines

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Include docstrings for functions and classes
- Maintain comprehensive error handling

### Documentation
- Keep README files updated
- Document all configuration options
- Include usage examples
- Maintain API documentation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Guidelines
- Write meaningful commit messages
- Update documentation as needed
- Add tests for new features
- Maintain code quality standards

## Support

- Create an issue for bugs or feature requests
- Check existing issues before creating new ones
- Provide detailed information when reporting issues

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- LangChain team for the excellent framework
- OpenFGA team for the authorization system
- Gradio team for the UI framework
- All contributors to the project

## Roadmap

- [ ] Add more LLM providers
- [ ] Implement additional RAG variations
- [ ] Create more UI examples
- [ ] Add comprehensive testing
- [ ] Improve documentation