import streamlit as st
import re
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
import nltk

# Ensure the 'punkt' tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)


@st.cache_data # Caching the function to avoid re-training on every run
def build_models(text, max_n=3):
    """
    Build multiple n-gram models from the text.
    This function is cached, so it only runs when the input text changes.
    """
    words = word_tokenize(text)
    models = {}
    for n in range(2, max_n + 1):
        model = defaultdict(Counter)
        for i in range(len(words) - n + 1):
            context = tuple(words[i:i + n - 1])
            next_word = words[i + n - 1]
            model[context][next_word] += 1
        models[n] = model
    return models

def get_autocomplete_suggestions(models, input_words, max_n=3, top_k=5):
    if not input_words:
        return [], ""
    
    partial_word = input_words[-1]
    context_words = input_words[:-1]

    if not context_words:
        return [], "Cannot autocomplete the first word. Please provide more context."

    for n in range(max_n, 1, -1):
        model = models.get(n)
        context_length = n - 1

        if len(context_words) < context_length:
            continue

        context = tuple(context_words[-context_length:])
        
        if model and context in model:
            predictions = model[context]
            suggestions = {word: count for word, count in predictions.items() if word.startswith(partial_word)}
            
            if not suggestions:
                continue

            total_occurrences = sum(predictions.values())
            sorted_suggestions = sorted(suggestions.items(), key=lambda item: item[1], reverse=True)
            
            result = []
            for word, count in sorted_suggestions[:top_k]:
                probability = (count / total_occurrences) * 100
                result.append((word, f"{probability:.2f}%"))
            return result, f"Found in {n}-gram model with context `{context}`"
            
    return [], "No autocomplete suggestions found."

def get_next_word_predictions(models, input_words, max_n=3, top_k=5):
    if not input_words:
        return [], "Enter text to get next-word predictions."

    for n in range(max_n, 1, -1):
        context_length = n - 1
        model = models.get(n)

        if len(input_words) < context_length:
            continue

        context = tuple(input_words[-context_length:])

        if model and context in model:
            predictions = model[context]
            total_occurrences = sum(predictions.values())
            top_predictions = predictions.most_common(top_k)
            
            result = []
            for word, count in top_predictions:
                probability = (count / total_occurrences) * 100
                result.append((word, f"{probability:.2f}%"))
            return result, f"Found in {n}-gram model with context `{context}`"

    return [], "No next-word predictions found."


st.set_page_config(page_title="N-Gram Word Predictor", layout="wide")

st.title("N-Gram Word Predictor & Autocomplete")
st.write("This app uses N-gram models to predict the next word and suggest autocompletions. Train the model on your own text and see the predictions live!")

# Initialize session state
if 'models' not in st.session_state:
    st.session_state.models = None

# Default training data
default_training_data = open("train.txt", "r").read()

st.sidebar.header("Model Configuration")
training_text_input = st.sidebar.text_area("1. Enter Training Text", default_training_data, height=250)
max_n_value = st.sidebar.slider("Select Max N-gram size (e.g., 3 for trigrams)", 2, 5, 3)

if st.sidebar.button("Train Model"):
    with st.spinner("Cleaning text and building models... This might take a moment."):
        # Clean the text
        cleaned_text = re.sub(r'\s+', ' ', training_text_input).strip().lower()
        if cleaned_text:
            st.session_state.models = build_models(cleaned_text, max_n_value)
            st.sidebar.success("✅ Model trained successfully!")
        else:
            st.sidebar.error("Training text cannot be empty.")

st.divider()

if st.session_state.models:
    st.header("🔮 Get Predictions")
    user_input = st.text_input("Enter text here:", "the lazy")

    input_words = user_input.lower().strip().split()

    if input_words:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Autocomplete Suggestions")
            st.write(f"Completing the last word: `{input_words[-1]}`")
            auto_suggestions, auto_message = get_autocomplete_suggestions(st.session_state.models, input_words, max_n_value)
            st.info(auto_message)
            if auto_suggestions:
                for word, prob in auto_suggestions:
                    st.write(f"**{word}** ({prob})")
        
        with col2:
            st.subheader("Next-Word Predictions")
            st.write(f"Predicting word after: `{' '.join(input_words)}`")
            next_predictions, next_message = get_next_word_predictions(st.session_state.models, input_words, max_n_value)
            st.info(next_message)
            if next_predictions:
                for word, prob in next_predictions:
                    st.write(f"**{word}** ({prob})")
else:
    st.info("Please train a model using the sidebar to begin.")
