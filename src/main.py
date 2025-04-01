# File: /ai-normalization-project/ai-normalization-project/src/main.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.utils.normalization import load_model, match_word
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Initializing AI normalization process...")
    model = load_model()

    # Example usage of word matching
    word = "my example"
    word_list = ["sample", "test", "example", "demo"]
    match, score = match_word(word, word_list, model)
    print(f"Best match for '{word}' is '{match}' with confidence {score:.2f}")

if __name__ == "__main__":
    main()