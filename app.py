import os
import streamlit as st
import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Get API token from environment variable
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not API_TOKEN:
    st.error("Missing HUGGINGFACE_API_TOKEN environment variable. Please set it in your .env file.")

# Use models that are actually text generation models, not fill-mask models
ALTERNATIVE_MODELS = {
    "FLAN-T5 Small": "google/flan-t5-small",  # General purpose but works well
    "DialoGPT-Med": "microsoft/DialoGPT-medium", # Good for dialogue/text generation
    "GPT-2 Med": "gpt2-medium" # Reliable text generation
}

def extract_text_from_pdf(pdf_file):
    """Extract text content from uploaded PDF file"""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def summarize_with_openai(text):
    """Alternative: Use OpenAI for summarization if available"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.warning("OpenAI API key not found. Falling back to built-in summarization.")
        return summarize_with_fallback(text)
    
    try:
        import openai
        client = openai.OpenAI(api_key=openai_api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a medical professional assistant. Summarize the clinical report into a concise summary that preserves all critical medical information. Include sections for summary, key findings, and recommendations."},
                {"role": "user", "content": f"Clinical Report:\n{text}"}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error with OpenAI API: {str(e)}")
        return summarize_with_fallback(text)

def summarize_with_fallback(text):
    """Built-in basic summarization as fallback"""
    # Extract key sentences based on basic NLP techniques
    import re
    from collections import Counter
    
    # Split text into sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    
    # Calculate word frequency
    words = re.findall(r'\w+', text.lower())
    word_freq = Counter(words)
    
    # Remove common medical stop words
    stop_words = {'the', 'and', 'of', 'to', 'a', 'in', 'that', 'is', 'was', 'for', 'on', 'with'}
    for word in stop_words:
        if word in word_freq:
            del word_freq[word]
    
    # Score sentences based on word frequency
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        for word in re.findall(r'\w+', sentence.lower()):
            if word in word_freq:
                if i not in sentence_scores:
                    sentence_scores[i] = 0
                sentence_scores[i] += word_freq[word]
    
    # Get top 30% of sentences
    top_sentences_count = max(3, int(len(sentences) * 0.3))
    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:top_sentences_count]
    top_sentences = sorted(top_sentences, key=lambda x: x[0])
    
    # Construct summary
    summary = [sentences[i] for i, _ in top_sentences]
    
    # Generate sections
    output = "# SUMMARY\n"
    output += " ".join(summary[:2]) + "\n\n"
    
    output += "# KEY FINDINGS\n"
    # Extract sentences with medical terms
    medical_terms = ['diagnosis', 'symptoms', 'treatment', 'medication', 'condition', 'patient', 'test', 'results']
    findings = []
    for sentence in summary:
        if any(term in sentence.lower() for term in medical_terms):
            findings.append(f"- {sentence}")
    output += "\n".join(findings) + "\n\n"
    
    output += "# RECOMMENDATIONS\n"
    output += "Based on the clinical report, the following recommendations are suggested:\n"
    output += "- Further review by a qualified healthcare professional is recommended\n"
    output += "- Consider additional testing based on the findings\n"
    
    return output

def summarize_with_huggingface(text, model_name):
    """Attempt to use Hugging Face for summarization"""
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
    
    # Truncate text if too long - most API endpoints have token limits
    max_chars = 4000
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    
    prompt = f"""
    Summarize the following clinical report:
    
    {text}
    
    Format:
    # SUMMARY
    # KEY FINDINGS
    # RECOMMENDATIONS
    """
    
    # Different models require different parameters
    if "flan-t5" in model_name.lower():
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 500,
                "temperature": 0.7
            }
        }
    else:
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.7,
                "do_sample": True
            }
        }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            if "generated_text" in result[0]:
                return result[0]["generated_text"]
            else:
                return str(result[0])
        elif isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"]
        else:
            return str(result)
    except Exception as e:
        st.error(f"Error during API call: {str(e)}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            st.error(f"API response: {e.response.text}")
        
        # Try OpenAI if Hugging Face fails
        if os.getenv("OPENAI_API_KEY"):
            st.warning("Trying OpenAI as fallback...")
            return summarize_with_openai(text)
        else:
            st.warning("Using built-in summarization as fallback...")
            return summarize_with_fallback(text)

def create_ui():
    """Create Streamlit user interface"""
    st.set_page_config(page_title="MedSum - Clinical Report Summarization", layout="wide")
    
    # Header
    st.title("MedSum: Medical Report Summarization")
    st.write("""
    MedSum automatically generates concise, accurate summaries of clinical documentation 
    while maintaining critical medical information integrity.
    """)
    
    # Sidebar for model selection and settings
    st.sidebar.title("Settings")
    
    summarization_method = st.sidebar.radio(
        "Select summarization method:",
        ["Hugging Face API", "OpenAI API (if available)", "Built-in (basic)"]
    )
    
    model_name = None
    if summarization_method == "Hugging Face API":
        selected_model = st.sidebar.selectbox(
            "Select Hugging Face model:",
            list(ALTERNATIVE_MODELS.keys())
        )
        model_name = ALTERNATIVE_MODELS[selected_model]
    
    # File upload or text input
    input_method = st.radio(
        "Select input method:",
        ("Upload PDF", "Enter Text")
    )
    
    text_to_summarize = None
    
    if input_method == "Upload PDF":
        uploaded_file = st.file_uploader("Upload clinical report (PDF)", type="pdf")
        if uploaded_file is not None:
            text_to_summarize = extract_text_from_pdf(uploaded_file)
            if text_to_summarize:
                st.success("PDF successfully processed")
                with st.expander("View extracted text"):
                    st.text_area("Extracted Text", text_to_summarize, height=300)
    else:
        text_to_summarize = st.text_area("Enter clinical report text:", height=300, placeholder="Enter or paste your clinical report text here...")
    
    # Process button
    if text_to_summarize and st.button("Generate Summary"):
        with st.spinner("Generating medical summary..."):
            if summarization_method == "Hugging Face API" and model_name:
                summary = summarize_with_huggingface(text_to_summarize, model_name)
            elif summarization_method == "OpenAI API (if available)" and os.getenv("OPENAI_API_KEY"):
                summary = summarize_with_openai(text_to_summarize)
            else:
                summary = summarize_with_fallback(text_to_summarize)
            
            if summary:
                st.subheader("Generated Summary")
                st.markdown(summary)
                
                # Add download option for the summary
                st.download_button(
                    "Download Summary",
                    summary,
                    file_name="medsum_report.md",
                    mime="text/markdown"
                )
            else:
                st.error("Failed to generate summary. Please try again.")
    
    # Add disclaimer
    st.sidebar.markdown("""
    ## Disclaimer
    This tool is intended to assist medical professionals and should not replace 
    professional medical judgment. Always review the generated summaries for accuracy.
    """)
    
    # Add information about the models
    st.sidebar.markdown("""
    ## About MedSum
    MedSum uses language models to summarize clinical reports.
    The application offers multiple summarization methods depending on your API access.
    """)

if __name__ == "__main__":
    create_ui()