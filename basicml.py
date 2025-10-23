import re
import random
import PyPDF2
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
import nltk


nltk.download('punkt', quiet=True)

# --- STEP 1: Read and clean text ---
def load_text(path):
    """Extract text from a PDF or text file and clean it for processing."""
    print(f"\n---  Step 1: Loading and Cleaning Text from '{path}' ---")
    if path.endswith('.pdf'):
        print("File type detected: PDF")
        try:
            reader = PyPDF2.PdfReader(path)
            text = "".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            return ""
    else:
        print(f"File type detected: Text file {path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: The file '{path}' was not found.")
            return ""

    print(f"Raw text snippet (first 100 chars): '{text[:100]}...'")
    print(f"Total characters read: {len(text)}")

    # Clean text: remove extra whitespace and convert to lowercase
    print("\nCleaning text: removing extra whitespace and converting to lowercase...")
    text = re.sub(r'\s+', ' ', text).strip().lower()
    print(f" Cleaned text snippet (first 100 chars): '{text[:100]}...'")
    print("---  Step 1 Complete ---")
    return text

# --- STEP 2: Build multiple n-gram models ---
def build_models(text, max_n=3):
    """
    Build multiple n-gram models, from n=2 (bigram) up to max_n.
    Returns a dictionary of models, e.g., {2: bigram_model, 3: trigram_model}.
    """
    print(f"\n--- Step 2: Building N-gram Models up to n={max_n} ---")
    words = word_tokenize(text)
    print(f"Total words (tokens) found: {len(words)}")
    
    models = {}
    for n in range(2, max_n + 1):
        print(f"\n-- Building {n}-gram model (context size: {n-1} word(s)) --")
        model = defaultdict(Counter)
        # Sliding window to create contexts and count next words
        for i in range(len(words) - n + 1):
            context = tuple(words[i:i + n - 1])
            next_word = words[i + n - 1]
            model[context][next_word] += 1
        
        print(f"Finished {n}-gram model with {len(model)} unique contexts.")
        models[n] = model

    print("\n--- Step 2 Complete ---")
    return models, words

# --- STEP 3A: Autocomplete the current word ---
def get_autocomplete_suggestions(models, input_words, max_n=3, top_k=5):
    """Provide autocomplete suggestions for the last (potentially partial) word."""
    if not input_words:
        return

    print(f"\n--- Autocompleting: '{' '.join(input_words)}' ---")
    
    partial_word = input_words[-1]
    context_words = input_words[:-1]

    if not context_words:
        print("Cannot autocomplete the first word. Please provide more context.")
        return

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

            print(f" Context {context} found in the {n}-gram model with completions for '{partial_word}'!")
            total_occurrences = sum(predictions.values())
            
            sorted_suggestions = sorted(suggestions.items(), key=lambda item: item[1], reverse=True)
            
            print(f"\nTop {min(top_k, len(sorted_suggestions))} autocomplete suggestions:")
            for word, count in sorted_suggestions[:top_k]:
                probability = (count / total_occurrences) * 100
                print(f"  → '{word}' ({probability:.2f}% probability)")
            return

    print("\nCould not find any autocomplete suggestions.")


# --- STEP 3B: Predict the next word ---
def get_next_word_predictions(models, input_words, max_n=3, top_k=5):
    """
    Predict the next word using a backoff strategy and show probabilities.
    Starts with the largest n-gram model and 'backs off' to smaller ones if no context is found.
    """
    print(f"\n---  Predicting word after: '{' '.join(input_words)}' ---")

    for n in range(max_n, 1, -1):
        context_length = n - 1
        model = models.get(n)

        if len(input_words) < context_length:
            continue

        context = tuple(input_words[-context_length:])

        if model and context in model:
            print(f" Context {context} found in the {n}-gram model!")
            predictions = model[context]
            
            total_occurrences = sum(predictions.values())
            print(f"This context appeared a total of {total_occurrences} time(s).")
            
            top_predictions = predictions.most_common(top_k)
            
            print(f"\nTop {min(top_k, len(top_predictions))} next-word predictions:")
            for word, count in top_predictions:
                probability = (count / total_occurrences) * 100
                print(f"  → '{word}' ({probability:.2f}% probability)")
            return

    print("\nCould not find a next-word prediction for the given text.")

# --- MAIN ---
if __name__ == "__main__":
    print("=============================================")
    print(" N-Gram Word Predictor & Autocomplete ")
    print("=============================================")
    print("A 'train.txt' file has been created for demonstration.")

    path = "train.txt"
    max_n_value = 3 

    text = load_text(path)
    if not text:
        print("Text could not be loaded. Exiting.")
        exit()
        
    models, words = build_models(text, max_n=max_n_value)

    print(f"\n Models trained successfully on {len(words)} words.")
    print("You can now enter text to get autocomplete and next-word predictions.")
    print("=============================================\n")

    while True:
        user_input = input("Enter text (or 'exit'): ").lower().strip()
        if user_input == 'exit':
            print("Exiting program. Goodbye!")
            break
        
        input_words = user_input.split()
        
        if not input_words:
            print(" Please enter some text.")
            continue

        get_autocomplete_suggestions(models, input_words, max_n=max_n_value)
        get_next_word_predictions(models, input_words, max_n=max_n_value)
        print("\n---------------------------------------------\n")

