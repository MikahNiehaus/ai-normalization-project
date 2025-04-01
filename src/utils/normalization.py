from sentence_transformers import SentenceTransformer, util
import torch
import os

# Check and log the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

LOCAL_MODEL_DIR = "./local_model"

def min_max_scaling(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

def z_score_normalization(data):
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std_dev = variance ** 0.5
    return [(x - mean) / std_dev for x in data]

class WordMatcher:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", threshold=0.3):
        """
        Initialize the WordMatcher class.

        Args:
            model_name (str): The name of the pre-trained model to use.
            threshold (float): The similarity threshold for a valid match.
        """
        self.threshold = threshold
        self.model = self._load_model(model_name)

    def _load_model(self, model_name):
        """Load the pre-trained model locally or download it if not available."""
        if not os.path.exists(LOCAL_MODEL_DIR) or not os.listdir(LOCAL_MODEL_DIR):
            print(f"Model not found locally. Downloading model '{model_name}' to '{LOCAL_MODEL_DIR}'...")
            model = SentenceTransformer(model_name, device=device)
            model.save(LOCAL_MODEL_DIR)
        else:
            print(f"Loading model from local directory '{LOCAL_MODEL_DIR}'...")
            model = SentenceTransformer(LOCAL_MODEL_DIR, device=device)
        return model

    def match(self, word, word_list):
        """
        Match a word or phrase against a list of words or phrases.

        Args:
            word (str): The input word or phrase to match.
            word_list (list): The list of words or phrases to match against.

        Returns:
            tuple: The best match and its score, or (None, score) if no match is above the threshold.
        """
        print(f"Matching word: '{word}' against list: {word_list}")
        word_embedding = self.model.encode(word, convert_to_tensor=True)
        list_embeddings = self.model.encode(word_list, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(word_embedding, list_embeddings)[0]
        best_match_idx = torch.argmax(similarities).item()
        best_match_score = similarities[best_match_idx].item()
        print(f"Scores: {similarities.tolist()}")
        print(f"Best match: '{word_list[best_match_idx]}' with score: {best_match_score}")
        if best_match_score < self.threshold:
            print(f"No match found (score below threshold: {self.threshold})")
            return None, best_match_score
        return word_list[best_match_idx], best_match_score

# Train the model with a local dataset (if needed)
def train_model(dataset_path, model_name="sentence-transformers/all-MiniLM-L6-v2", output_dir="./trained_model"):
    print("Training is not supported for this model directly. Use fine-tuning methods provided by sentence-transformers.")