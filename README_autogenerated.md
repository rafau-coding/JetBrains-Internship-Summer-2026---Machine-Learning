# Pure NumPy Word2Vec Implementation

This repository contains a from-scratch implementation of the Word2Vec algorithm, written entirely in pure Python and NumPy. It was developed as a submission for the JetBrains Machine Learning Internship task.

## 🧠 Architecture Overview

This implementation utilizes the **Skip-Gram with Negative Sampling (SGNS)** architecture. 

Instead of a standard Softmax layer—which is computationally prohibitive over large vocabularies—this model treats the context prediction as a series of binary classification tasks. It maximizes the dot product of true target-context word pairs while minimizing the dot product of random (negative) pairs.

**Key Features:**
* Pure NumPy optimization loop (Forward pass, Loss calculation, Backpropagation, Parameter updates).
* Custom data pipeline for vocabulary building and positive/negative pair generation.
* Unigram distribution-based negative sampling (scaled by the standard 0.75 power).

## 🚀 Getting Started

### Prerequisites

The only dependency required to run this model is NumPy.
\`\`\`bash
pip install numpy
\`\`\`

### Running the Code
1. Clone the repository.
2. Ensure you have your dataset ready. By default, the script expects a text file named `dataset.txt` in the root directory. 
3. Execute the training script:
\`\`\`bash
python word2vec.py
\`\`\`

## 📂 Dataset Information
*(Note to applicant: Fill this out based on what you actually use!)*
The model is currently configured to train on the [Text8 corpus / Project Gutenberg Book Title]. The data loader handles basic tokenization, but for production environments, a more robust preprocessing pipeline (lowercasing, punctuation removal) is recommended.

## 🔮 Possible Optimizations (Future Work)
If this were to be scaled further, the following optimizations would be implemented:
1. **Subsampling of frequent words:** Probabilistically discarding highly frequent, low-information words (like "the" or "a") to speed up training and improve rare word representations.
2. **Dynamic Window Sizes:** Randomly shrinking the context window during training to give more weight to immediately adjacent words.
3. **Learning Rate Decay:** Implementing a linearly decreasing learning rate over time. Currently, the model uses a fixed learning rate, but gradually reducing it as training progresses helps the model converge more smoothly and prevents it from overshooting the optimal weights in the final epochs.