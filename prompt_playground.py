import streamlit as st
import threading
import time
from langchain_ollama import OllamaLLM
from st_copy_to_clipboard import st_copy_to_clipboard

# ------------------------ Configuration ------------------------

# Mapping of models to their maximum token limits
MODEL_MAX_TOKENS = {
    "phi4:latest": 4096,
    "llama3.2:latest": 4000,
    "mistral-nemo:latest": 128000,
    # Add more models and their max tokens as needed
}

# List of available Ollama models
AVAILABLE_MODELS = list(MODEL_MAX_TOKENS.keys())

# ------------------------ Streamlit Setup ------------------------

st.set_page_config(
    page_title="Krish - The Model Comparison",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔥 Krish - Compare Models | Optimize Prompt")
st.markdown("---")

# ------------------------ Clear Button ------------------------

col1, col2 = st.columns([6, 1])
with col2:
    if st.button("Clear 🗑️"):
        if "outputs" in st.session_state:
            del st.session_state["outputs"]
        st.success("Session cleared successfully!")

# ------------------------ Sidebar Inputs ------------------------

with st.sidebar:
    st.header("🔧 Configuration")
    st.markdown("### 🌐 Base URL")
    base_url = st.text_input(
        label="Ollama API Base URL",
        value="http://localhost:11434",
        placeholder="Enter the base URL for the Ollama API...",
        help="Specify the base URL where your Ollama server is hosted (e.g., http://localhost:11434).",
    )
    st.markdown("### 📚 Select Models to Compare")
    model_options = ["Select a model"] + AVAILABLE_MODELS

    model1 = st.selectbox(
        label="Select Model 1",
        options=model_options,
        index=0,
        key="model1",
        help="Choose the first Ollama model to use for generating responses.",
    )
    if model1 != "Select a model":
        temp1 = st.slider(
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.01,
            key="temp1",
            help="Higher values means the model will take more risks. Try 0.9 for creative apps, 0 for deterministic answers.",
        )
        max_tokens1 = st.slider(
            label="Max Output Tokens",
            min_value=512,
            max_value=MODEL_MAX_TOKENS.get(model1, 4096),
            value=int(MODEL_MAX_TOKENS.get(model1, 4096) * 0.1),
            step=1,
            key="max_tokens1",
            help=f"Maximum allowed: {MODEL_MAX_TOKENS.get(model1, 4096)} (used as 'num_predict').",
        )
    else:
        temp1 = st.slider(
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.01,
            disabled=True,
            key="temp1_disabled",
            help="Select a model to enable this slider.",
        )
        max_tokens1 = st.slider(
            label="Max Output Tokens",
            min_value=512,
            max_value=4096,
            value=500,
            step=1,
            disabled=True,
            key="max_tokens1_disabled",
            help="Select a model to enable this slider.",
        )

    model2 = st.selectbox(
        label="Select Model 2",
        options=["Select a model"] + [m for m in AVAILABLE_MODELS if m != model1],
        index=0,
        key="model2",
        help="Choose the second Ollama model to use for generating responses.",
    )
    if model2 != "Select a model":
        temp2 = st.slider(
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.01,
            key="temp2",
            help="Higher values means the model will take more risks.",
        )
        max_tokens2 = st.slider(
            label="Max Output Tokens",
            min_value=512,
            max_value=MODEL_MAX_TOKENS.get(model2, 4096),
            value=int(MODEL_MAX_TOKENS.get(model2, 4096) * 0.1),
            step=1,
            key="max_tokens2",
            help=f"Maximum allowed: {MODEL_MAX_TOKENS.get(model2, 4096)}",
        )
    else:
        temp2 = st.slider(
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.01,
            disabled=True,
            key="temp2_disabled",
            help="Select a model to enable this slider.",
        )
        max_tokens2 = st.slider(
            label="Max Output Tokens",
            min_value=512,
            max_value=4096,
            value=500,
            step=1,
            disabled=True,
            key="max_tokens2_disabled",
            help="Select a model to enable this slider.",
        )

    model3 = st.selectbox(
        label="Select Model 3",
        options=["Select a model"]
        + [m for m in AVAILABLE_MODELS if m not in [model1, model2]],
        index=0,
        key="model3",
        help="Choose the third Ollama model to use for generating responses.",
    )
    if model3 != "Select a model":
        temp3 = st.slider(
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.01,
            key="temp3",
            help="Higher values means the model will take more risks.",
        )
        max_tokens3 = st.slider(
            label="Max Output Tokens",
            min_value=512,
            max_value=4096,
            value=int(MODEL_MAX_TOKENS.get(model3, 4096) * 0.1),
            step=1,
            key="max_tokens3",
            help=f"Maximum allowed: {MODEL_MAX_TOKENS.get(model3, 4096)}",
        )
    else:
        temp3 = st.slider(
            label="Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.01,
            disabled=True,
            key="temp3_disabled",
            help="Select a model to enable this slider.",
        )
        max_tokens3 = st.slider(
            label="Max Output Tokens",
            min_value=512,
            max_value=4096,
            value=500,
            step=1,
            disabled=True,
            key="max_tokens3_disabled",
            help="Select a model to enable this slider.",
        )

# ------------------------ Input Fields in Main Area ------------------------

st.header("📝 Input Fields")
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        prompt = st.text_area(
            label="📄 Prompt",
            placeholder="Enter your prompt here...",
            height=300,
            help="Provide the initial prompt to guide the models.",
        )
    with col2:
        context = st.text_area(
            label="📚 Context",
            placeholder="Enter additional context here (optional)...",
            height=300,
            help="Provide any additional context that might help the models generate better responses.",
        )
    question = st.text_area(
        label="❓ Question",
        placeholder="Enter your question here...",
        height=200,
        help="Ask a specific question related to the prompt and context.",
    )

# ------------------------ Run Button ------------------------

run_button = st.button("Run 🏃‍♂️", type="primary")

# ------------------------ Main Logic with Streaming UI ------------------------

if run_button:
    with st.spinner("Processing your request..."):
        try:
            if not base_url:
                st.error("Please enter a valid Ollama API Base URL.")
            else:
                # Collect chosen models
                selected_models = []
                configurations = {}
                if model1 != "Select a model":
                    selected_models.append(model1)
                    configurations[model1] = {
                        "temperature": temp1,
                        "num_predict": max_tokens1,
                    }
                if model2 != "Select a model":
                    selected_models.append(model2)
                    configurations[model2] = {
                        "temperature": temp2,
                        "num_predict": max_tokens2,
                    }
                if model3 != "Select a model":
                    selected_models.append(model3)
                    configurations[model3] = {
                        "temperature": temp3,
                        "num_predict": max_tokens3,
                    }

                if not selected_models:
                    st.error("Please select at least one model to compare.")
                else:
                    # Combine the prompt, context, and question into a single input
                    full_input = f"Prompt: {prompt}\n\nContext: {context}\n\nQuestion: {question}"
                    # Shared dict to hold final text for each model
                    responses = {model: "" for model in selected_models}

                    # Create columns, one per model
                    num_models = len(selected_models)
                    output_cols = st.columns(num_models)

                    # One placeholder per model to display the streaming text
                    placeholders = {}
                    for idx, model in enumerate(selected_models):
                        with output_cols[idx]:
                            placeholders[model] = st.empty()

                    # Worker function: streams tokens into 'responses'
                    def worker(model, config):
                        llm = OllamaLLM(
                            model=model,
                            temperature=config["temperature"],
                            num_predict=config["num_predict"],
                            base_url=base_url,
                        )
                        for token in llm.stream(input=full_input):
                            responses[model] += token
                            time.sleep(0.01)

                    # Start threads
                    threads = []
                    for model in selected_models:
                        t = threading.Thread(
                            target=worker, args=(model, configurations[model])
                        )
                        t.start()
                        threads.append(t)

                    # Polling loop: update placeholders with the partial text
                    while any(t.is_alive() for t in threads):
                        for idx, model in enumerate(selected_models):
                            emoji = "🟢" if idx == 0 else "🔵" if idx == 1 else "🟠"
                            placeholders[model].markdown(
                                f"""
                                <h3>{emoji} {model}</h3>
                                <div style="
                                    border: 1px solid #444444;
                                    border-radius: 8px;
                                    padding: 15px;
                                    background-color: #2c2c2c;
                                    color: #ffffff;
                                    margin-bottom: 10px;">
                                    {responses[model]}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        time.sleep(0.2)

                    # Ensure all threads are finished
                    for t in threads:
                        t.join()

                    # After all threads finish
                    for idx, model in enumerate(selected_models):
                        # Decide which emoji to display
                        emoji = "🟢" if idx == 0 else ("🔵" if idx == 1 else "🟠")
                        # Clear the placeholder
                        placeholders[model].empty()
                        # Render final text and the copy button in one container
                        with placeholders[model].container():
                            st.markdown(
                                f"""
                                <h3>{emoji} {model}</h3>
                                <div style="
                                    border: 1px solid #444444;
                                    border-radius: 8px;
                                    padding: 15px;
                                    background-color: #2c2c2c;
                                    color: #ffffff;
                                    margin-bottom: 10px;">
                                    {responses[model]}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            # Copy button right below the final response
                            st_copy_to_clipboard(responses[model], key=f"copy_{model}")

                    st.session_state["outputs"] = responses

        except Exception as e:
            st.error(f"Error processing request: {str(e)}")

# ------------------------ End of Script ------------------------

st.markdown("---")
st.markdown("Made with ❤️ using Ollama Models")
