import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader

# Paths
LOCAL_MODEL_DIR = "./local_model"
DATASET_DIR = "./data"
TRAINED_MODEL_DIR = "./trained_model"

# Dataset and model configuration
DATASET_NAME = "quora"  # Change this to another dataset if needed
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def download_dataset():
    """Download the dataset if it does not exist locally."""
    dataset_path = os.path.join(DATASET_DIR, DATASET_NAME)
    if not os.path.exists(dataset_path):
        print(f"Downloading dataset '{DATASET_NAME}'...")
        dataset = load_dataset(DATASET_NAME, trust_remote_code=True)  # Added trust_remote_code=True
        os.makedirs(DATASET_DIR, exist_ok=True)
        dataset.save_to_disk(dataset_path)
    else:
        print(f"Dataset '{DATASET_NAME}' already exists locally.")
    return os.path.join(DATASET_DIR, DATASET_NAME)

def load_training_data(dataset_path):
    """Load the training data from the dataset."""
    print(f"Loading dataset from '{dataset_path}'...")
    dataset = load_dataset("quora", split="train", trust_remote_code=True)  # Added trust_remote_code=True
    examples = []
    for row in dataset:
        examples.append(InputExample(texts=[row["questions"]["text"][0], row["questions"]["text"][1]], label=float(row["is_duplicate"])))
    return examples

def train_model():
    """Train the model using the dataset."""
    # Ensure dataset is downloaded
    dataset_path = download_dataset()

    # Load training data
    train_examples = load_training_data(dataset_path)

    # Load the model
    print(f"Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    # Prepare the training data loader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Define the loss function
    train_loss = losses.CosineSimilarityLoss(model)

    # Train the model
    print("Starting training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,  # Adjust epochs as needed
        warmup_steps=100,
        output_path=TRAINED_MODEL_DIR
    )
    print(f"Model trained and saved to '{TRAINED_MODEL_DIR}'.")

if __name__ == "__main__":
    train_model()
