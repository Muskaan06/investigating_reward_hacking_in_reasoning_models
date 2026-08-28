import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import os

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
INPUT_FILE = "free_sentences.txt"
OUTPUT_FILE = "final_token_activations_free.npy"

# Use GPU if available
DEVICE = "mps" if torch.mps.is_available() else "cpu"

print(f"Loading model on {DEVICE}...")

# ============================================================
# Load tokenizer and model
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
)

model.to(DEVICE)
model.eval()

# ============================================================
# Read statements
# ============================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    statements = [line.strip() for line in f if line.strip()]

print(f"Found {len(statements)} statements.")

# ============================================================
# Extract final-layer, final-token activations
# ============================================================

activations = []

with torch.no_grad():

    for i, statement in enumerate(statements):

        # Tokenize the statement
        inputs = tokenizer(
            statement,
            return_tensors="pt",
            truncation=True
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        # Forward pass
        outputs = model(
            **inputs,
            output_hidden_states=True
        )

        # ----------------------------------------------------
        # hidden_states:
        #
        # hidden_states[0] = embedding output
        # hidden_states[1] = layer 1
        # ...
        # hidden_states[-1] = final transformer layer
        #
        # Shape:
        # [batch_size, sequence_length, hidden_size]
        # ----------------------------------------------------

        final_layer = outputs.hidden_states[-1]

        # Final token of the sentence
        final_token_activation = final_layer[0, -1, :]

        # Move to CPU and convert to numpy
        activation = final_token_activation.float().cpu().numpy()

        activations.append(activation)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(statements)}")

# ============================================================
# Convert to array
# ============================================================

activations = np.stack(activations)

print("\nFinished.")
print("Activation shape:", activations.shape)

# ============================================================
# Save
# ============================================================

np.save(OUTPUT_FILE, activations)

print(f"Saved activations to: {OUTPUT_FILE}")