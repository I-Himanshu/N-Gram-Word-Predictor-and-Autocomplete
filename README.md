# N-Gram Word Predictor & Autocomplete Streamlit App

This project is an interactive web application built with Streamlit that demonstrates the concepts of N-gram models for natural language processing. It can read text input, build predictive models, and then offer real-time next-word predictions and autocomplete suggestions based on user input.


## ✨ Features

* **Dynamic N-Gram Modeling**: Builds multiple n-gram models (bigram, trigram, etc.) from the source text.

* **Next-Word Prediction**: Uses a backoff strategy to predict the most likely next word in a sequence, complete with probability scores.

* **Autocomplete Suggestions**: Provides suggestions to complete the current word being typed based on the learned context.

* **Interactive UI**: A simple and clean user interface powered by Streamlit for easy interaction.

## ⚙️ How It Works

The application tokenizes the text from a given source file and builds a series of N-gram models. An N-gram is a contiguous sequence of *n* items (in this case, words) from a sample of text.

* **Model Building**: For an n-gram model, it creates a dictionary where each key is a tuple of `n-1` words (the "context"), and the value is a counter of all the words that have appeared immediately after that context.

* **Prediction**: To predict the next word, it looks for the current context in the largest model (e.g., trigram). If found, it returns the most frequent next words.

* **Backoff Strategy**: If the context isn't found, it "backs off" to a smaller model (e.g., bigram) and tries again. This makes the predictions more robust.


## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

* Python 3.8+

* `pip` for package management

### Installation & Running the App

1. **Clone the repository:**

   ```
   git clone [https://github.com/your-username/n-gram-word-predictor.git](https://github.com/your-username/n-gram-word-predictor.git)
   cd n-gram-word-predictor
   
   ```

2. **Create and activate a virtual environment (recommended):**

   ```
   # For Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # For macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   
   ```

3. **Install the required dependencies:**
   The `requirements.txt` file contains all the necessary packages.

   ```
   pip install -r requirements.txt
   
   ```

4. **Run the Streamlit application:**

   ```
   streamlit run app.py
   
   ```

   Your web browser should open a new tab with the running application.

## Project Structure

```
.
├── app.py              # The main Streamlit application script
├── train.txt           # A sample text file for training
├── requirements.txt    # Project dependencies
└── README.md           # This file

```

## 🛠️ Built With

* [Python](https://www.python.org/)

* [Streamlit](https://streamlit.io/) - The core framework for the web app

* [NLTK](https://www.nltk.org/) - For text tokenization

* [PyPDF2](https://pypdf2.readthedocs.io/) - For extracting text from PDF files