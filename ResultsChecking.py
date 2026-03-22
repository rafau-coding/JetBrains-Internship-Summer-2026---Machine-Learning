import numpy as np
import json
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE



class Word2VecSGNS:
    def __init__(self, vocab_size, embed_dim, learning_rate=0.01):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lr = learning_rate

        # Initialize target word embeddings (W) and context word embeddings (C)
        # It's common practice to initialize W randomly and C to zeros
        self.W = np.random.uniform(-0.5 / embed_dim, 0.5 / embed_dim, (vocab_size, embed_dim))
        self.C = np.zeros((vocab_size, embed_dim))

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


def visualize_embeddings_interactive(filepath="my_word2vec", num_words=400):
    """
    Loads saved Word2Vec embeddings and creates a highly interactive,
    zoomable, and scrollable 2D t-SNE plot using Plotly.
    """
    print(f"Loading model files from '{filepath}'...")

    try:
        with open(f"{filepath}_vocab.json", "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}_vocab.json. Did you run the training script?")
        return

    idx2word = {int(k): v for k, v in vocab_data["idx2word"].items()}

    weights = np.load(f"{filepath}_weights.npz")
    W = weights['W']

    limit = min(num_words, len(idx2word))
    words = [idx2word[i] for i in range(limit)]
    vectors = W[:limit]

    print(f"Running t-SNE on {limit} words...")
    perplexity = min(30, limit - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    vectors_2d = tsne.fit_transform(vectors)

    # 1. Package the data into a Pandas DataFrame (Plotly loves DataFrames)
    df = pd.DataFrame({
        'Word': words,
        'X': vectors_2d[:, 0],
        'Y': vectors_2d[:, 1]
    })

    # 2. Build the interactive scatter plot
    print("Opening interactive plot in your web browser...")
    fig = px.scatter(
        df,
        x='X',
        y='Y',
        text='Word',  # This puts the word permanently next to the dot
        hover_name='Word',  # This highlights the word if you hover over it
        title=f"Interactive t-SNE Visualization of {limit} Word Embeddings"
    )

    # 3. Format the aesthetics and layout
    fig.update_traces(
        textposition='top center',
        textfont=dict(size=11, color='#333333'),
        marker=dict(size=10, color='royalblue', opacity=0.7, line=dict(width=1, color='white'))
    )

    fig.update_layout(
        template="plotly_white",
        width=1400,  # Base width of the canvas
        height=900,  # Base height of the canvas
        dragmode='pan',  # Defaults your mouse click to panning/dragging
        hovermode='closest',
        title_font=dict(size=24)
    )

    # 4. Render the plot (This will pop open a new tab in Chrome/Firefox/Safari)
    fig.show()

# --- Checking ---

if __name__ == "__main__":
    # To load the model in a new script:
    loaded_model, loaded_word2idx, loaded_idx2word = Word2VecSGNS.load_model("my_word2vec")

    print(loaded_idx2word.values())
    # brudnopis: "dutch", "congress", "big", "house", "home",

    list1 = ["go", "went"]
    list2 = ["mainstream", "lillian"]
    list3 = [list1, list2]
    # Check the embedding for 'fox'
    for l in list3:
        for x in l:
            if x in loaded_idx2word.values():
                print(f"Embedding for '{x}':\n",loaded_model.W[loaded_word2idx[x]][:5], "...")
        if l[0] in loaded_idx2word.values() and l[1] in loaded_idx2word.values():
            print("Dot product: ", np.dot(loaded_model.W[loaded_word2idx[l[0]]],loaded_model.W[loaded_word2idx[l[1]]]))
        print("")

    visualize_embeddings_interactive(filepath="my_word2vec", num_words=400)