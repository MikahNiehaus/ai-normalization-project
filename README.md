# AI Normalization Project

This project implements AI-based word matching and normalization using `sentence-transformers`. It includes functionality to match a word or phrase against a list of words or phrases and determine the best match or if no match exists. The project also supports fine-tuning the model using a dataset.

## Features
- Match a word or phrase against a list of words or phrases.
- Determine the best match or return `None` if no match exists.
- Fine-tune the model using the `quora` dataset or other datasets.

---

## Setup Instructions

### 1. Create a Virtual Environment
Create and activate a virtual environment to isolate dependencies:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Requirements
Install all required dependencies:
```bash
pip install -r requirements.txt
```

### 3. Install PyTorch
Install the correct version of PyTorch for your system. Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) for the appropriate command. For example:
- **CUDA 11.8**:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
- **CPU-only**:
  ```bash
  pip install torch torchvision torchaudio
  ```

---

## Usage Instructions

### 1. Run the Main Script
To use the word-matching functionality, run the main script:
```bash
python src/main.py
```
This will:
- Load the pre-trained model.
- Match an example word (`"my example"`) against a list of words (`["sample", "test", "example", "demo"]`).
- Print the best match and its confidence score.

### 2. Run Unit Tests
To verify the functionality, run the unit tests:
```bash
python src/tests/test_normalization.py
```
This will:
- Test various scenarios, such as matching a word, handling misspellings, and determining no match.

### 3. Train the Model
To fine-tune the model using the `quora` dataset:
```bash
python src/train_model.py
```
This will:
- Download the `quora` dataset if it does not exist locally.
- Fine-tune the model using the dataset.
- Save the trained model to the `./trained_model` directory.

---

## How It Works

### Word Matching
The `WordMatcher` class in `src/utils/normalization.py` provides the word-matching functionality. It:
1. Loads a pre-trained `sentence-transformers` model.
2. Matches a word or phrase against a list of words or phrases.
3. Returns the best match and its score, or `None` if no match exceeds the similarity threshold.

#### Example Usage
```python
from src.utils.normalization import WordMatcher

# Initialize the WordMatcher
matcher = WordMatcher(threshold=0.3)

# Match a word against a list
word = "example"
word_list = ["sample", "test", "example", "demo"]
match, score = matcher.match(word, word_list)

print(f"Best match: {match}, Score: {score}")
```

### Training
The `train_model.py` script fine-tunes the model using a dataset. By default, it uses the `quora` dataset, but you can modify the script to use other datasets.

---

## Project Structure

```
ai-normalization-project
├── src
│   ├── main.py                  # Main script for word matching
│   ├── train_model.py           # Script for fine-tuning the model
│   ├── tests
│   │   └── test_normalization.py # Unit tests for word matching
│   └── utils
│       └── normalization.py     # WordMatcher class and utility functions
├── requirements.txt             # List of dependencies
└── README.md                    # Project documentation
```

---

## Notes
- Ensure you have a compatible GPU and CUDA installed for faster training and inference.
- Modify the `threshold` parameter in the `WordMatcher` class to adjust the sensitivity of word matching.
- Replace the `quora` dataset in `train_model.py` with another dataset if needed.

---
