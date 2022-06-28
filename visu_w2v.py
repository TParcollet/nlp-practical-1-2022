import numpy as np
import pandas as pd
import torch
import sys

from sklearn.manifold import TSNE
import plotly.graph_objects as go

def get_top_similar(word, topN = 10, vocab = None, embeddings_norm = None):
    word_id = vocab[word]
    if word_id == 0:
        print("Out of vocabulary word")
        return

    word_vec = embeddings_norm[word_id]

    # We must compute the distance between all words (simple np.matmul !)
    #
    #

    topN_dict = {}
    for sim_word_id in topN_ids:
        sim_word = vocab.lookup_token(sim_word_id)
        topN_dict[sim_word] = dists[sim_word_id]
    return topN_dict

# Load the params
folder = "weights/cbow_WikiText2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(f"{folder}/model.pt", map_location=device)
vocab = torch.load(f"{folder}/vocab.pt")

# embedding from first model layer
#
#

# normalization
#
#
embeddings_norm = embeddings / norms

# get embeddings
embeddings_df = pd.DataFrame(embeddings)

# t-SNE transform
tsne = TSNE(n_components=2)
embeddings_df_trans = tsne.fit_transform(embeddings_df)
embeddings_df_trans = pd.DataFrame(embeddings_df_trans)

# get token order
embeddings_df_trans.index = vocab.get_itos()

# if token is a number
is_numeric = embeddings_df_trans.index.str.isnumeric()

color = np.where(is_numeric, "green", "black")
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=embeddings_df_trans[0],
        y=embeddings_df_trans[1],
        mode="text",
        text=embeddings_df_trans.index,
        textposition="middle center",
        textfont=dict(color=color),
    )
)
fig.write_html("word2vec_visualization.html")

for word, sim in get_top_similar("france", vocab = vocab, embeddings_norm = embeddings_norm).items():
    print("{}: {:.3f}".format(word, sim))

# Get a word to play with from the vocab
#
#

# Arithmeticaly play with it in the vector space (+, -, x)
#

# Normalise it (Example is given)
emb4_norm = (emb4 ** 2).sum() ** (1 / 2)
emb4 = emb4 / emb4_norm


# Compute the distance (matmul) and reshape before, :p
#
#

# Get the top5 corresponding words
top5 = np.argsort(...)[:5]

for word_id in top5:
    print("{}: {:.3f}".format(vocab.lookup_token(word_id), dists[word_id]))
