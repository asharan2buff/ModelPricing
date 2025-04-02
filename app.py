import streamlit as st
import pandas as pd
import tiktoken
import io

# Set page configuration
st.set_page_config(
    page_title="AI Model Price Calculator",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
        background-color: #F9FAFB;
    }
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    h1 {
        color: #6366F1;
        font-weight: 700;
    }
    h2 {
        color: #6366F1;
        font-weight: 600;
        margin-top: 2rem;
    }
    .stButton>button {
        background: linear-gradient(to right, #6366F1, #8B5CF6);
        color: white;
        font-weight: 600;
    }
    .result-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-top: 2rem;
    }
    .total-price {
        font-size: 1.5rem;
        font-weight: 700;
        color: #10B981;
    }
</style>
""", unsafe_allow_html=True)

# Define model pricing data (same as React app)
model_data = {
    "llama": [
        {"name": "Llama 3 Text (Up to 3B)", "price": 0.06},
        {"name": "Llama 3 Text (8B) - LITE", "price": 0.10},
        {"name": "Llama 3 Text (8B) - TURBO", "price": 0.18},
        {"name": "Llama 3 Text (8B) - REFERENCE", "price": 0.20},
        {"name": "Llama 3 Vision (11B)", "price": 0.18},
        {"name": "Llama 3 Text (70B) - LITE", "price": 0.54},
        {"name": "Llama 3 Text (70B) - TURBO", "price": 0.88},
        {"name": "Llama 3 Text (70B) - REFERENCE", "price": 0.90},
        {"name": "Llama 3 Vision (90B)", "price": 1.20},
        {"name": "Llama 3 Text (405B) - TURBO", "price": 3.50}
    ],
    "deepseek": [
        {"name": "DeepSeek-V3", "price": 1.25},
        {"name": "DeepSeek-R1", "inputPrice": 3.00, "outputPrice": 7.00},
        {"name": "Deepseek-R1-Distill-Llama-70B", "price": 2.00},
        {"name": "Deepseek-R1-Distill-Qwen-14B", "price": 1.60},
        {"name": "Deepseek-R1-Distill-Qwen-1.5B", "price": 0.18},
        {"name": "DeepSeek LLM Chat 67B", "price": 0.90}
    ],
    "qwen": [
        {"name": "Qwen 2 72B", "price": 0.90},
        {"name": "Qwen 2-VL-72B", "price": 1.20},
        {"name": "Qwen 2.5 7B", "price": 0.30},
        {"name": "Qwen 2.5 14B", "price": 0.80},
        {"name": "Qwen 2.5 72B", "price": 1.20},
        {"name": "Qwen 2.5 Coder 32B", "price": 0.80},
        {"name": "Qwen QwQ 32B Preview", "price": 1.20}
    ],
    "other": [
        {"name": "Up to 4B", "price": 0.10},
        {"name": "4.1B - 8B", "price": 0.20},
        {"name": "8.1B - 21B", "price": 0.30},
        {"name": "21.1B - 41B", "price": 0.80},
        {"name": "41.1B - 80B", "price": 0.90},
        {"name": "80.1B - 110B", "price": 1.80}
    ],
    "moe": [
        {"name": "Up to 56B total parameters", "price": 0.60},
        {"name": "56.1B - 176B total parameters", "price": 1.20},
        {"name": "176.1B - 480B total parameters", "price": 2.40}
    ],
    "flux": [
        {"name": "FLUX.1 [dev]", "price": 0.025},
        {"name": "FLUX.1 [dev] lora", "price": 0.035},
        {"name": "FLUX.1 [schnell]", "price": 0.0027},
        {"name": "FLUX1.1 [pro]", "price": 0.04},
        {"name": "FLUX.1 [pro]", "price": 0.05},
        {"name": "FLUX.1 Canny [dev]", "price": 0.025},
        {"name": "FLUX.1 Depth [dev]", "price": 0.025},
        {"name": "FLUX.1 Redux [dev]", "price": 0.025}
    ]
}

# Function to count tokens using tiktoken
def count_tokens(text, model="cl100k_base"):
    """Count tokens in a text string using tiktoken"""
    try:
        encoding = tiktoken.get_encoding(model)
        token_count = len(encoding.encode(text))
        return token_count
    except Exception as e:
        st.error(f"Error counting tokens: {e}")
        return 0

# Function to process CSV and count tokens from the "notes" column
def process_csv(df):
    """Process CSV and count tokens from 'notes' column"""
    if 'notes' not in df.columns:
        st.error("The CSV file does not contain a 'notes' column")
        return 0
    
    # Count tokens in each row of the notes column
    total_tokens = 0
    for note in df['notes'].fillna(""):  # Replace NaN with empty string
        if isinstance(note, str):
            total_tokens += count_tokens(note)
    
    return total_tokens

# App Header
st.title("AI Model Price Calculator")
st.markdown("Calculate the cost of using AI models based on token count")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["CSV Token Calculator", "Prompt Token Calculator"])

# Initialize session state for storing results
if 'csv_tokens' not in st.session_state:
    st.session_state.csv_tokens = 0
if 'prompt_tokens' not in st.session_state:
    st.session_state.prompt_tokens = 0
if 'output_tokens' not in st.session_state:
    st.session_state.output_tokens = 0
if 'rag_tokens' not in st.session_state:
    st.session_state.rag_tokens = 0

# Tab 1: CSV Upload and Token Count
with tab1:
    st.header("Upload CSV with Notes")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            st.success(f"File uploaded successfully. Shape: {df.shape}")
            
            # Show first few rows
            with st.expander("Preview CSV Data"):
                st.dataframe(df.head())
            
            # Count tokens in the notes column
            token_count = process_csv(df)
            st.session_state.csv_tokens = token_count
            
            # Display token count
            st.markdown(f"""
            <div class="result-card"style="color: black;>
                <h3>CSV Token Count Result</h3>
                <p>Total tokens in 'notes' column: <b>{token_count:,}</b></p>
                <p>Millions of tokens: <b>{token_count/1000000:.6f}</b></p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error processing CSV: {e}")

# Tab 2: Prompt Token Calculator
with tab2:
    st.header("Prompt Token Calculator")
    
    # Input for prompt text
    prompt_text = st.text_area("Enter your prompt text", height=200)
    
    # Count prompt tokens when content changes
    if prompt_text:
        prompt_token_count = count_tokens(prompt_text)
        st.session_state.prompt_tokens = prompt_token_count
        st.markdown(f"Prompt tokens: **{prompt_token_count:,}**")
    
    # Output tokens (for classification tasks)
    output_tokens = st.number_input("Estimated output tokens", min_value=0, value=100)
    st.session_state.output_tokens = output_tokens
    
    # Optional RAG tokens
    with st.expander("RAG (Retrieval Augmented Generation)"):
        rag_tokens = st.number_input("Additional RAG tokens", min_value=0, value=0)
        st.session_state.rag_tokens = rag_tokens
    
    # Display total token count for prompt calculation
    total_prompt_tokens = st.session_state.prompt_tokens + st.session_state.output_tokens + st.session_state.rag_tokens
    st.markdown(f"""
    <div class="result-card" style="color: black;">
        <h3>Prompt Token Count Result</h3>
        <p>Prompt tokens: <b>{st.session_state.prompt_tokens:,}</b></p>
        <p>Output tokens: <b>{st.session_state.output_tokens:,}</b></p>
        <p>RAG tokens: <b>{st.session_state.rag_tokens:,}</b></p>
        <p>Total tokens: <b>{total_prompt_tokens:,}</b></p>
        <p>Millions of tokens: <b>{total_prompt_tokens/1000000:.6f}</b></p>
    </div>
    """, unsafe_allow_html=True)

# Price Calculation Section
st.markdown("---")
st.header("Price Calculation")

# Model selection
col1, col2 = st.columns(2)

with col1:
    # Model category selection
    category = st.selectbox(
        "Model Category",
        options=list(model_data.keys()),
        format_func=lambda x: {
            "llama": "Llama 3.3, 3.2, 3.1, 3 Models",
            "deepseek": "DeepSeek Models",
            "qwen": "Qwen Models",
            "other": "All Other Chat/Language/Code Models",
            "moe": "Mixture-of-Experts Models",
            "flux": "FLUX Image Models"
        }[x]
    )

with col2:
    # Model name selection based on category
    model_names = [model["name"] for model in model_data[category]]
    model_name = st.selectbox("Model", options=model_names)

# Get the selected model data
selected_model = next((model for model in model_data[category] if model["name"] == model_name), None)

# Token input section depends on the model type
if selected_model:
    has_separate_pricing = "inputPrice" in selected_model and "outputPrice" in selected_model
    is_flux_model = category == "flux"
    
    st.markdown("### Token Count Input")
    
    # Choose token source
    token_source = st.radio(
        "Token Source",
        ["CSV Tokens", "Prompt Tokens", "Manual Input"],
        horizontal=True
    )
    
    if token_source == "CSV Tokens":
        # Use tokens from CSV
        token_count = st.session_state.csv_tokens / 1000000  # Convert to millions
        st.info(f"Using {st.session_state.csv_tokens:,} tokens ({token_count:.6f} million) from CSV")
        
        # Separate input/output for models like DeepSeek-R1
        if has_separate_pricing:
            col1, col2 = st.columns(2)
            with col1:
                input_ratio = st.slider("Input Token Ratio", 0.0, 1.0, 0.75, 0.01)
            with col2:
                output_ratio = 1 - input_ratio
                st.markdown(f"Output Token Ratio: **{output_ratio:.2f}**")
            
            input_tokens = token_count * input_ratio
            output_tokens = token_count * output_ratio
            
    elif token_source == "Prompt Tokens":
        # Use tokens from prompt calculator
        token_count = total_prompt_tokens / 1000000  # Convert to millions
        st.info(f"Using {total_prompt_tokens:,} tokens ({token_count:.6f} million) from Prompt Calculator")
        
        # Separate input/output for models like DeepSeek-R1
        if has_separate_pricing:
            input_tokens = st.session_state.prompt_tokens / 1000000  # Convert to millions
            output_tokens = (st.session_state.output_tokens + st.session_state.rag_tokens) / 1000000  # Convert to millions
            st.markdown(f"Input: {input_tokens:.6f} million, Output: {output_tokens:.6f} million")
            
    else:  # Manual Input
        # Manual token count input
        if has_separate_pricing:
            col1, col2 = st.columns(2)
            with col1:
                input_tokens = st.number_input(
                    "Input Tokens (in millions)",
                    min_value=0.0,
                    value=0.75,
                    step=0.1,
                    format="%.6f"
                )
            with col2:
                output_tokens = st.number_input(
                    "Output Tokens (in millions)",
                    min_value=0.0,
                    value=0.25,
                    step=0.1,
                    format="%.6f"
                )
            token_count = input_tokens + output_tokens
        elif is_flux_model:
            token_count = st.number_input(
                "Number of Megapixels (in millions)",
                min_value=0.1,
                value=1.0,
                step=0.1,
                format="%.2f"
            )
        else:
            token_count = st.number_input(
                "Number of Tokens (in millions)",
                min_value=0.1,
                value=1.0,
                step=0.1,
                format="%.6f"
            )
    
    # Calculate button
    if st.button("Calculate Price", use_container_width=True):
        details = []
        total_price = 0
        
        if has_separate_pricing:
            # For models with separate input/output pricing (like DeepSeek-R1)
            input_cost = input_tokens * selected_model["inputPrice"]
            output_cost = output_tokens * selected_model["outputPrice"]
            total_price = input_cost + output_cost
            
            details.append(f"Input: {input_tokens:.6f} million tokens × ${selected_model['inputPrice']:.2f} = ${input_cost:.2f}")
            details.append(f"Output: {output_tokens:.6f} million tokens × ${selected_model['outputPrice']:.2f} = ${output_cost:.2f}")
            details.append(f"Total: ${input_cost:.2f} + ${output_cost:.2f} = ${total_price:.2f}")
            
        elif is_flux_model:
            # For FLUX image models, pricing is per megapixel
            total_price = token_count * selected_model["price"]
            details.append(f"{token_count:.2f} million megapixels × ${selected_model['price']:.4f} per megapixel = ${total_price:.2f}")
            
        else:
            # Standard token-based pricing
            total_price = token_count * selected_model["price"]
            details.append(f"{token_count:.6f} million tokens × ${selected_model['price']:.2f} per million tokens = ${total_price:.2f}")
        
        # Display results
        st.markdown("""
        <div class="result-card" style="color: black;">
            <h5>Price Calculation Result</h5>
        """, unsafe_allow_html=True)
        
        for detail in details:
            st.markdown(f"<p>{detail}</p>", unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 1rem; margin-top: 1rem; 
                      background-color: #F9FAFB; border-radius: 0.5rem; border: 1px solid #E5E7EB;">
                <span style="font-weight: 600; color: #1F2937;">Total Price:</span>
                <span class="total-price">${total_price:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("© 2025 AI Model Price Calculator. Created with Streamlit.")
