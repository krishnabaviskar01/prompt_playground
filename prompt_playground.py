import streamlit as st
from langchain_ollama import ChatOllama
from st_copy_to_clipboard import st_copy_to_clipboard
import markdown  # To convert markdown text to HTML

# ------------------------ Configuration ------------------------

# Mapping of models to their maximum token limits
MODEL_MAX_TOKENS = {
    'mistral:latest': 32000,
    'qwen2:latest': 128000,
    'llama3.2:3b-instruct-q8_0': 128000,
    'llama3.2:latest': 128000,
    'llama3-groq-tool-use:latest': 8192, 
    'mistral-nemo:latest': 128000,
    'deepseek-coder:1.3b-instruct': 16000,
    'llama3.1:latest': 128000,
    'codellama:latest': 100000,
    'llama3:latest': 128000,
    # Add more models and their max tokens as needed
}

# List of available Ollama models
AVAILABLE_MODELS = list(MODEL_MAX_TOKENS.keys())

# ------------------------ Streamlit Setup ------------------------

# Configure the Streamlit page
st.set_page_config(
    page_title="🔥 Krish - The Model Comparison",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title of the application
st.title("🔥 Krish - Compare Models | Optimize Prompt")

# Add a horizontal line separator
st.markdown("---")

# ------------------------ Clear Button ------------------------

# Add a "Clear" button at the top right to reset the session
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("Clear 🗑️"):
        # Clear the output from the session state
        if 'outputs' in st.session_state:
            del st.session_state['outputs']
        # Optionally, clear other session states or inputs if necessary
        st.success("Session cleared successfully!")

# ------------------------ Sidebar Inputs ------------------------

with st.sidebar:
    st.header("🔧 Configuration")

    st.markdown("### 🌐 Base URL")
    # Input for Base URL
    base_url = st.text_input(
        label="Ollama API Base URL",
        value="http://localhost:11434",  # Default value
        placeholder="Enter the base URL for the Ollama API...",
        help="Specify the base URL where your Ollama server is hosted (e.g., http://localhost:11434)."
    )

    st.markdown("### 📚 Select Models to Compare")
    # Prepend a placeholder option for blank selection
    model_options = ["Select a model"] + AVAILABLE_MODELS

    # Allow users to select three distinct models
    model1 = st.selectbox(
        label="Select Model 1",
        options=model_options,
        index=0,  # Default to "Select a model"
        key="model1",
        help="Choose the first Ollama model to use for generating responses."
    )

    # Temperature slider and Max Tokens for Model 1 (only if a model is selected)
    if model1 != "Select a model":
        temp1 = st.slider(
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.01,
            key="temp1",
            help=f"Temperature for ({model1}): Adjusts the randomness of responses. Lower values (e.g., 0.1) make outputs more focused and deterministic, ideal for precise tasks. Higher values (e.g., 0.8 or 1.0) make responses more diverse and creative. Set to 0 for completely deterministic behavior."
        )

        max_tokens1 = st.slider(
            label="Max Output Tokens",
            min_value=50,
            max_value=MODEL_MAX_TOKENS.get(model1, 4096),
            value=int(MODEL_MAX_TOKENS.get(model1, 4096) * 0.1),  # Default to 10% of max tokens
            step=50,
            key="max_tokens1",
            help=f"Set the maximum number of tokens for {model1}'s response. Maximum allowed: {MODEL_MAX_TOKENS.get(model1, 4096)}"
        )
    else:
        # Placeholder sliders when no model is selected
        temp1 = st.slider(
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.01,
            disabled=True,
            key="temp1_disabled",
            help="Select a model to enable this slider."
        )

        max_tokens1 = st.slider(
            label="Max Output Tokens",
            min_value=50,
            max_value=4096,
            value=500,
            step=50,
            disabled=True,
            key="max_tokens1_disabled",
            help="Select a model to enable this slider."
        )

    model2 = st.selectbox(
        label="Select Model 2",
        options=["Select a model"] + [model for model in AVAILABLE_MODELS if model != model1],
        index=0,
        key="model2",
        help="Choose the second Ollama model to use for generating responses."
    )

    # Temperature slider and Max Tokens for Model 2 (only if a model is selected)
    if model2 != "Select a model":
        temp2 = st.slider(
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.01,
            key="temp2",
            help=f"Temperature for ({model2}): Adjusts the randomness of responses. Lower values (e.g., 0.1) make outputs more focused and deterministic, ideal for precise tasks. Higher values (e.g., 0.8 or 1.0) make responses more diverse and creative. Set to 0 for completely deterministic behavior."
        )

        max_tokens2 = st.slider(
            label="Max Output Tokens",
            min_value=50,
            max_value=MODEL_MAX_TOKENS.get(model2, 4096),
            value=int(MODEL_MAX_TOKENS.get(model2, 4096) * 0.1),
            step=50,
            key="max_tokens2",
            help=f"Set the maximum number of tokens for {model2}'s response. Maximum allowed: {MODEL_MAX_TOKENS.get(model2, 4096)}"
        )
    else:
        # Placeholder sliders when no model is selected
        temp2 = st.slider(
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.01,
            disabled=True,
            key="temp2_disabled",
            help="Select a model to enable this slider."
        )

        max_tokens2 = st.slider(
            label="Max Output Tokens",
            min_value=50,
            max_value=4096,
            value=500,
            step=50,
            disabled=True,
            key="max_tokens2_disabled",
            help="Select a model to enable this slider."
        )

    model3 = st.selectbox(
        label="Select Model 3",
        options=["Select a model"] + [model for model in AVAILABLE_MODELS if model != model1 and model != model2],
        index=0,
        key="model3",
        help="Choose the third Ollama model to use for generating responses."
    )

    # Temperature slider and Max Tokens for Model 3 (only if a model is selected)
    if model3 != "Select a model":
        temp3 = st.slider(
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.01,
            key="temp3",
            help=f"Temperature for ({model3}): Adjusts the randomness of responses. Lower values (e.g., 0.1) make outputs more focused and deterministic, ideal for precise tasks. Higher values (e.g., 0.8 or 1.0) make responses more diverse and creative. Set to 0 for completely deterministic behavior."
        )

        max_tokens3 = st.slider(
            label="Max Output Tokens",
            min_value=50,
            max_value=MODEL_MAX_TOKENS.get(model3, 4096),
            value=int(MODEL_MAX_TOKENS.get(model3, 4096) * 0.1),
            step=50,
            key="max_tokens3",
            help=f"Set the maximum number of tokens for {model3}'s response. Maximum allowed: {MODEL_MAX_TOKENS.get(model3, 4096)}"
        )
    else:
        # Placeholder sliders when no model is selected
        temp3 = st.slider(
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.01,
            disabled=True,
            key="temp3_disabled",
            help="Select a model to enable this slider."
        )

        max_tokens3 = st.slider(
            label="Max Output Tokens",
            min_value=50,
            max_value=4096,
            value=500,
            step=50,
            disabled=True,
            key="max_tokens3_disabled",
            help="Select a model to enable this slider."
        )

# ------------------------ Input Fields in Main Area ------------------------

st.header("📝 Input Fields")

with st.container():
    # Organize input fields in two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        # Text area for Prompt
        prompt = st.text_area(
            label="📄 Prompt",
            placeholder="Enter your prompt here...",
            height=300,
            help="Provide the initial prompt to guide the models."
        )

    with col2:
        # Text area for Context
        context = st.text_area(
            label="📚 Context",
            placeholder="Enter additional context here (optional)...",
            height=300,
            help="Provide any additional context that might help the models generate better responses."
        )

    # Full-width text area for Question
    question = st.text_area(
        label="❓ Question",
        placeholder="Enter your question here...",
        height=200,
        help="Ask a specific question related to the prompt and context."
    )

# ------------------------ Run Button ------------------------

# Place the Run button below the input fields
run_button = st.button("Run 🏃‍♂️", type="primary")

# ------------------------ Main Logic ------------------------

if run_button:
    with st.spinner("Processing your request..."):
        try:
            # Validate Base URL
            if not base_url:
                st.error("Please enter a valid Ollama API Base URL.")
            else:
                # Collect selected models and their configurations
                selected_models = []
                configurations = {}

                if model1 != "Select a model":
                    selected_models.append(model1)
                    configurations[model1] = {
                        'temperature': temp1,
                        'max_tokens': max_tokens1
                    }

                if model2 != "Select a model":
                    selected_models.append(model2)
                    configurations[model2] = {
                        'temperature': temp2,
                        'max_tokens': max_tokens2
                    }

                if model3 != "Select a model":
                    selected_models.append(model3)
                    configurations[model3] = {
                        'temperature': temp3,
                        'max_tokens': max_tokens3
                    }

                # Ensure at least one model is selected
                if not selected_models:
                    st.error("Please select at least one model to compare.")
                else:
                    # Combine the prompt, context, and question into a single input
                    full_input = f"Prompt: {prompt}\n\nContext: {context}\n\nQuestion: {question}"

                    # Initialize ChatOllama clients for each selected model with their respective temperatures and max_tokens
                    responses = {}
                    for model in selected_models:
                        llm = ChatOllama(
                            model=model,
                            temperature=configurations[model]['temperature'],
                            max_tokens=configurations[model]['max_tokens'],
                            base_url=base_url
                        )
                        response = llm.invoke(input=full_input)
                        responses[model] = response.content

                    # Store the responses in the session state
                    st.session_state['outputs'] = responses

        except Exception as e:
            # Display any errors that occur during processing
            st.error(f"Error processing request: {str(e)}")

# ------------------------ Display Outputs ------------------------

if 'outputs' in st.session_state:
    outputs = st.session_state['outputs']
    
    st.header("📄 Output Comparison")
    
    # Create columns based on the number of selected models
    num_models = len(outputs)
    if num_models == 1:
        cols = st.columns(1)
    elif num_models == 2:
        cols = st.columns(2)
    else:
        cols = st.columns(3)
    
    for idx, (model, content) in enumerate(outputs.items()):
        with cols[idx]:
            emoji = "🟢" if idx == 0 else "🔵" if idx == 1 else "🟠"
            st.subheader(f"{emoji} {model}")
            if content:
                # Convert markdown content to HTML
                html_content = markdown.markdown(content)

                # Display the content inside a styled box using HTML and CSS with dark theme
                st.markdown(f"""
                    <div style="
                        border: 1px solid #444444;
                        border-radius: 8px;
                        padding: 15px;
                        background-color: #2c2c2c;
                        color: #ffffff;
                        margin-bottom: 10px;
                    ">
                        {html_content}
                    </div>
                """, unsafe_allow_html=True)

                # Add a copy button below the response
                st_copy_to_clipboard(content, key=f"copy_{idx}")  # Copies raw markdown
            else:
                st.write("No response.")

else:
    st.info("Provide your inputs above and click 'Run' to generate and compare responses from the selected models.")

# Add a footer
st.markdown("---")
st.markdown("Made with ❤️ using Ollama Models")
