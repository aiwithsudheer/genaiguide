import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


# Custom CSS for better styling
custom_css = """
.container {
    max-width: 1200px !important;
    margin: auto;
}
.chat-container {
    border-radius: 10px !important;
    border: 1px solid #e5e5e5 !important;
}
.model-controls {
    background-color: #f9f9f9;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #e5e5e5;
}
.message-input {
    border-top: 1px solid #e5e5e5;
    padding-top: 20px;
}
footer {display: none !important;}
"""


# Initialize LLM models using LangChain
def get_llm(model_name, temperature=0.7):
    models = {
        "GPT-4": ChatOpenAI(model_name="gpt-4", streaming=True, temperature=temperature),
        "GPT-3.5": ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True, temperature=temperature),
        "Claude": ChatAnthropic(model_name="claude-3-5-sonnet-20240620", streaming=True, temperature=temperature),
        "Gemini": ChatGoogleGenerativeAI(model="gemini-1.5-pro", streaming=True, temperature=temperature),
        "Mistral": ChatMistralAI(model="mistral-large-latest", streaming=True, temperature=temperature)
    }
    return models.get(model_name)

def process_message(message, selected_model, history, system_prompt=None, temperature=0.7):
    try:
        # Convert history to LangChain format
        history_langchain_format = []
        
        if system_prompt:
            history_langchain_format.append(SystemMessage(content=system_prompt))
        
        # Add existing conversation history
        for human, ai in history:
            history_langchain_format.append(HumanMessage(content=human))
            history_langchain_format.append(AIMessage(content=ai))
        
        # Add new message
        history_langchain_format.append(HumanMessage(content=message))
        
        # Get LLM
        llm = get_llm(selected_model, temperature)
        if llm is None:
            new_history = history + [(message, "Error: Model not found")]
            return new_history
            
        # Initialize new history with human message
        new_history = history + [(message, "")]
        
        # Stream response
        partial_message = ""
        for chunk in llm.stream(history_langchain_format):
            if hasattr(chunk, 'content'):
                content = chunk.content
            else:
                content = str(chunk)
            
            partial_message += content
            new_history[-1] = (message, partial_message)
            yield new_history
        
    except Exception as e:
        new_history = history + [(message, f"Error: {str(e)}")]
        yield new_history

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as chat_interface:
    with gr.Column(elem_classes="container"):
        gr.Markdown(
            """
            # 🤖 LangChain Chat Interface
            Chat with multiple AI models through a unified interface
            """
        )
        
        with gr.Row():
            with gr.Column(scale=7, elem_classes="chat-container"):
                chatbot = gr.Chatbot(
                    height=600,
                    show_copy_button=True,
                    render_markdown=True,
                    container=True,
                    elem_classes="chat-window"
                )
                
                with gr.Row(elem_classes="message-input"):
                    msg = gr.Textbox(
                        label="Your Message",
                        placeholder="Type your message here...",
                        lines=3,
                        scale=9
                    )
                    submit_btn = gr.Button("Send", scale=1, variant="primary")
            
            with gr.Column(scale=3, elem_classes="model-controls"):
                model_dropdown = gr.Dropdown(
                    choices=["GPT-4", "GPT-3.5", "Claude", "Gemini", "Mistral"],
                    value="GPT-3.5",
                    label="Select Model",
                    info="Choose your preferred LLM",
                    container=True
                )
                
                system_msg = gr.Textbox(
                    label="System Message",
                    placeholder="Set the behavior and context for the AI...",
                    lines=3,
                    info="This message sets the context and behavior for the entire conversation",
                    container=True
                )
                
                temperature = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more creative, Lower = more focused",
                    container=True
                )
                
                clear = gr.Button("Clear Chat", variant="secondary")
                
                gr.Markdown(
                    """
                    ### Tips
                    - Use system message to set AI behavior
                    - Adjust temperature for response variety
                    - Responses are streamed in real-time
                    """
                )
    
    def clear_conversation():
        return None
    
    # Event handlers
    msg.submit(
        fn=process_message,
        inputs=[msg, model_dropdown, chatbot, system_msg, temperature],
        outputs=chatbot,
        show_progress=False
    ).then(
        fn=lambda: gr.update(value=""),
        inputs=None,
        outputs=msg
    )
    
    submit_btn.click(
        fn=process_message,
        inputs=[msg, model_dropdown, chatbot, system_msg, temperature],
        outputs=chatbot,
        show_progress=False
    ).then(
        fn=lambda: gr.update(value=""),
        inputs=None,
        outputs=msg
    )
    
    clear.click(clear_conversation, None, chatbot)

# Launch the interface
if __name__ == "__main__":
   chat_interface.launch()
