import numpy as np
import random
import urllib.request
import zipfile
import os
from collections import Counter
import json

def download_and_read_text8(save_path="text8.zip"):
    """Downloads the Text8 dataset from Matt Mahoney's website if not present."""
    if not os.path.exists("text8"):
        print("Downloading Text8 dataset (approx. 31MB zip)...")
        url = "http://mattmahoney.net/dc/text8.zip"
        urllib.request.urlretrieve(url, save_path)

        print("Extracting dataset...")
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall()
        print("Extraction complete.")

    print("Reading corpus into memory...")
    with open("text8", "r", encoding="utf-8") as f:
        # Text8 is already lowercased and stripped of punctuation
        text = f.read()
    return text.split()


def preprocess_corpus(words, min_count=5, max_words=None):
    """
    Filters rare words and optionally truncates the corpus for faster testing.
    """
    print(f"Original corpus size: {len(words)} words")

    if max_words:
        words = words[:max_words]
        print(f"Truncated corpus to: {len(words)} words for demonstration.")

    # Count word frequencies
    word_counts = Counter(words)

    # Keep only words that appear at least 'min_count' times
    vocab = [word for word, count in word_counts.items() if count >= min_count]
    vocab_set = set(vocab)

    print(f"Vocabulary size after filtering (min_count={min_count}): {len(vocab)}")

    # Remove out-of-vocabulary words from the sequence
    filtered_corpus = [word for word in words if word in vocab_set]
    print(f"Final training corpus size: {len(filtered_corpus)} words")

    return filtered_corpus

def sigmoid(x):
    # Clip x to prevent overflow in np.exp
    x = np.clip(x, -10, 10)
    return 1 / (1 + np.exp(-x))


class Word2VecSGNS:
    def __init__(self, vocab_size, embed_dim, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lr = learning_rate

        # Initialize target word embeddings (W) and context word embeddings (C)
        # It's common practice to initialize W randomly and C to zeros
        self.W = np.random.uniform(-0.5 / embed_dim, 0.5 / embed_dim, (vocab_size, embed_dim))
        self.C = np.zeros((vocab_size, embed_dim))

    def train_step(self, target_idx, context_idx, negative_indices):
        """
        Performs one forward pass, gradient calculation, and weight update.
        """
        # Fetch embeddings
        v_w = self.W[target_idx]  # Shape: (embed_dim,)
        v_c = self.C[context_idx]  # Shape: (embed_dim,)
        v_n = self.C[negative_indices]  # Shape: (K, embed_dim)

        # Forward pass: compute scores
        score_c = np.dot(v_c, v_w)  # Scalar
        score_n = np.dot(v_n, v_w)  # Shape: (K,)

        # Compute probabilities
        prob_c = sigmoid(score_c)  # Scalar
        prob_n = sigmoid(score_n)  # Shape: (K,)

        # Compute Loss (Optional, mainly for tracking progress)
        # loss = -np.log(prob_c + 1e-8) - np.sum(np.log(1 - prob_n + 1e-8))

        # Calculate errors (Gradients w.r.t. dot products)
        # Based on the math: (sigma(x) - 1) for true context, sigma(x) for negatives
        err_c = prob_c - 1.0  # Scalar
        err_n = prob_n  # Shape: (K,)

        # Compute gradients for embeddings
        grad_v_c = err_c * v_w  # Shape: (embed_dim,)
        grad_v_n = np.outer(err_n, v_w)  # Shape: (K, embed_dim)

        # Gradient for target word is sum of gradients from context and negatives
        grad_v_w = err_c * v_c + np.dot(err_n, v_n)  # Shape: (embed_dim,)

        # Parameter Updates (Gradient Descent)
        self.W[target_idx] -= self.lr * grad_v_w
        self.C[context_idx] -= self.lr * grad_v_c
        self.C[negative_indices] -= self.lr * grad_v_n

        return grad_v_c  # Return for the unit test

    def compute_loss(self, target_idx, context_idx, negative_indices):
        """Calculates the scalar loss for gradient checking."""
        v_w = self.W[target_idx]
        v_c = self.C[context_idx]
        v_n = self.C[negative_indices]

        pos_loss = -np.log(sigmoid(np.dot(v_c, v_w)) + 1e-8)
        neg_loss = -np.sum(np.log(sigmoid(-np.dot(v_n, v_w)) + 1e-8))
        return pos_loss + neg_loss

    def check_gradients(self, target_idx, context_idx, negative_indices):
        """Verifying the analytical gradient numerically."""
        epsilon = 1e-5
        # mathematical justification -> Taylor Series of L(w+eps) and L(w-eps)
        # 1. Get analytical gradient from our math
        analytical_grad_c = self.train_step(target_idx, context_idx, negative_indices)

        # 2. Compute numerical gradient for the first element of context vector
        self.C[context_idx][0] += epsilon
        loss_plus = self.compute_loss(target_idx, context_idx, negative_indices)

        self.C[context_idx][0] -= 2 * epsilon  # Step back to -epsilon
        loss_minus = self.compute_loss(target_idx, context_idx, negative_indices)

        self.C[context_idx][0] += epsilon  # Reset weight to original

        numerical_grad_c0 = (loss_plus - loss_minus) / (2 * epsilon)

        # Compare the first element of our calculated gradient to the numerical one
        difference = abs(analytical_grad_c[0] - numerical_grad_c0)
        print(f"Gradient Check Diff (Should be < 1e-5): {difference:.8f}")

    def save_model(self, filepath, word2idx, idx2word):
        """Saves the model weights and vocabulary."""
        # 1. Save the NumPy matrices efficiently
        np.savez(f"{filepath}_weights.npz", W=self.W, C=self.C)

        # 2. Save the vocabulary and hyperparameters as JSON
        # Note: JSON keys must be strings, so we ensure idx2word keys are strings
        vocab_data = {
            "word2idx": word2idx,
            "idx2word": {str(k): v for k, v in idx2word.items()},
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim
        }
        with open(f"{filepath}_vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab_data, f)

        print(f"Model successfully saved to {filepath}_weights.npz and {filepath}_vocab.json")

    @classmethod
    def load_model(cls, filepath):
        """Loads a saved model and returns the model instance and vocab dictionaries."""
        # 1. Load vocabulary and hyperparameters
        with open(f"{filepath}_vocab.json", "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        # Reconstruct dictionaries (converting idx keys back to integers)
        word2idx = vocab_data["word2idx"]
        idx2word = {int(k): v for k, v in vocab_data["idx2word"].items()}

        # 2. Re-initialize the model with saved hyperparameters
        model = cls(vocab_size=vocab_data["vocab_size"],
                    embed_dim=vocab_data["embed_dim"])

        # 3. Load the weights into the model
        weights = np.load(f"{filepath}_weights.npz")
        model.W = weights['W']
        model.C = weights['C']

        print(f"Model successfully loaded from {filepath}_weights.npz")
        return model, word2idx, idx2word


# --- Helper functions for data preparation ---

def build_dataset(corpus, window_size=2, num_negatives=5):
    """
    Given a list of words, creates a vocabulary and generates training batches.
    """
    vocab = list(set(corpus))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}

    # Calculate word frequencies for negative sampling (often adjusted by power of 0.75)
    word_counts = Counter(corpus)
    total_words = sum(word_counts.values())
    unigram_dist = np.array([word_counts[idx2word[i]] for i in range(len(vocab))])
    unigram_dist = np.power(unigram_dist, 0.75)
    unigram_dist /= np.sum(unigram_dist)

    data_pairs = []
    # Generate positive pairs (target, context)
    for i, word in enumerate(corpus):
        target_idx = word2idx[word]
        # Get context window boundaries
        start = max(0, i - window_size)
        end = min(len(corpus), i + window_size + 1)

        for j in range(start, end):
            if i != j:
                context_idx = word2idx[corpus[j]]
                data_pairs.append((target_idx, context_idx))

    return vocab, word2idx, idx2word, unigram_dist, data_pairs


# --- Training Loop ---

if __name__ == "__main__":
    # 1. Load and preprocess the real data
    raw_words = download_and_read_text8()
    # raw_words -> list of strings

    # Note: We limit to 100,000 words here so the reviewer can actually
    # see it run locally in a reasonable time.
    corpus = preprocess_corpus(raw_words, min_count=5, max_words=100000)
    # corpus -> list of strings

    # 2. Hyperparameters
    window_size = 2
    # window_size -> integer
    num_negatives = 5
    # num_negatives -> integer
    embed_dim = 50  # Increased dimension for a larger dataset
    # embed_dim -> integer
    epochs = 5 # epochs -> integer
    learning_rate = 0.025  # Standard starting LR for Word2Vec
    # learning_rate -> float

    # 3. Build dataset pairs and vocab mapping
    print("Building training pairs...")
    vocab, word2idx, idx2word, unigram_dist, training_pairs = build_dataset(corpus, window_size)
    # vocab -> list of strings
    # word2idx -> dictionary (string to integer)
    # idx2word -> dictionary (integer to string)
    # unigram_dist -> 1D NumPy array of floats
    # training_pairs -> list of tuples (integer, integer)

    # 4. Initialize
    model = Word2VecSGNS(vocab_size=len(vocab), embed_dim=embed_dim, learning_rate=learning_rate)
    # model -> Word2VecSGNS class instance

    # 5. Run a Gradient Check before training
    print("\n--- Running Unit Test ---")
    dummy_negatives = np.random.choice(len(vocab), size=5, p=unigram_dist)
    # dummy_negatives -> 1D NumPy array of integers
    model.check_gradients(training_pairs[0][0], training_pairs[0][1], dummy_negatives)
    print("-------------------------\n")

    # 6. Train
    print(f"Starting training on {len(training_pairs)} pairs...")
    # epoch -> integer
    for epoch in range(epochs):
        random.shuffle(training_pairs)
        # step -> integer
        # target_idx -> integer
        # context_idx -> integer
        for step, (target_idx, context_idx) in enumerate(training_pairs):
            # negative_indices -> 1D NumPy array of integers
            negative_indices = np.random.choice(
                len(vocab), size=num_negatives, p=unigram_dist, replace=True
            )

            model.train_step(target_idx, context_idx, negative_indices)

            # Print progress so the reviewer knows it hasn't frozen
            if step % 10000 == 0 and step > 0:
                print(f"Epoch {epoch + 1}/{epochs} - Step {step}/{len(training_pairs)}")

        print(f"--- Epoch {epoch + 1}/{epochs} complete ---")

    print("\nTraining finished!")
    model.save_model("my_word2vec", word2idx, idx2word)

    # Quick sanity check
    test_word = {"one","two","three"}
    # test_word -> set of strings
    for x in test_word:
        if x in word2idx:
            print(f"Embedding for '{x}':\n", model.W[word2idx[x]][:5], "...")



    end = True # for readibility purposes


# SUMMARY
#
# This algorithm learns dense vector representations (embeddings) by trying to
# predict the context of a given word, based on the linguistic theory that
# "you shall know a word by the company it keeps."
#
# 1. Data Preparation: Windows and Pairs
# - Sliding Window: A window slides across the text. For window_size=2, it looks
#   at 2 words before and 2 words after a target center word.
# - Positive Pairs: It generates (target, context) pairs. E.g., for the phrase
#   "the quick brown fox", if "brown" is the target, positive pairs include
#   (brown, quick) and (brown, fox).
# - Negative Sampling Distribution: Word frequencies are calculated and raised to
#   the power of 0.75. This mathematical trick slightly boosts the probability of
#   picking rare words as "fake" context words, preventing common words (like "the")
#   from completely dominating the noise samples.
#
# 2. The Architecture: Two Embedding Matrices
# The model maintains two distinct matrices of size (vocab_size, embed_dim):
# - Target Matrix (W): Used when a word is the center/target word.
# - Context Matrix (C): Used when a word is the true surrounding/context word,
#   or a randomly drawn negative sample.
#   * Note: After training, C is usually discarded, and W is kept as the final
#     set of word embeddings.
#
# 3. The Forward Pass: Similarity via Dot Product
# - Lookups: For each step, the model fetches the target vector (v_w) from W,
#   the true context vector (v_c) from C, and K random negative vectors (v_n) from C.
# - Dot Products: It calculates similarity using dot products (e.g., v_c * v_w).
#   If the vectors are pointing in the same direction, the result is a large positive
#   number.
# - Sigmoid Activation: It squashes these raw dot product scores into probabilities
#   (between 0 and 1) using the sigmoid function: 1 / (1 + e^-x).
#
# 4. The Objective: Binary Classification
# Instead of a massive softmax over the entire vocabulary, Negative Sampling turns
# the problem into a set of independent binary classification tasks.
# - Goal 1: Push the probability of the true context word pair toward 1.
# - Goal 2: Push the probability of the K fake negative word pairs toward 0.
#
# 5. The Backward Pass: Gradient Calculation
# To minimize the loss, the algorithm calculates how much it should change the
# vectors (the gradients). The calculus simplifies beautifully here:
# - Error for the true context word: (predicted_probability - 1.0)
# - Error for the negative words: (predicted_probability - 0.0)
# These scalar errors are then multiplied by the respective vectors to determine
# the exact direction and magnitude needed to adjust the weights.
#
# 6. The Update: Gradient Descent
# Finally, the algorithm takes a step in the opposite direction of the gradient
# to update matrices W and C.
# - It pulls the target vector and the true context vector closer together in space.
# - It pushes the target vector and the negative vectors further apart.
# - The 'learning_rate' dictates how large of a step is taken.
#
# By repeating this over millions of pairs, words that share similar contexts
# naturally end up clustered close together in the multi-dimensional space.
#
