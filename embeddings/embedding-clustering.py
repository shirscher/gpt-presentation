#
# Does not work - see https://www.mlq.ai/fine-tuning-gpt-semantic-search-classification-regression/ for tutorial
#
import os
import pandas as pd
import numpy as np
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from transformers import GPT2TokenizerFast
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

openai.api_key = os.getenv("OPENAI_API_KEY")

input_datapath = 'Reviews.csv'  # This dataset includes 500k reviews 
print("Loading data...")
df = pd.read_csv(input_datapath, index_col=0)
print("Done!")
df = df[['Time', 'ProductId', 'UserId', 'Score', 'Summary', 'Text']]
df = df.dropna()
df['combined'] = "Title: " + df.Summary.str.strip() + "; Content: " + df.Text.str.strip()
df = df.sort_values('Time').tail(1_00) # Pick latest 300 reviews 
df.drop('Time', axis=1, inplace=True)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2") 
df['n_tokens'] = df.combined.apply(lambda x: len(tokenizer.encode(x))) #add number of tokens
df = df[df.n_tokens<8000].tail(1_00) # remove extra long text lines based on number of tokens
print("Calculating embeddings...")
df['ada_similarity'] = df.combined.apply(lambda x: get_embedding(x, engine='text-embedding-ada-002'))
print("Done!")

print("Calculating cosine similarity...")
matrix = np.array(df.ada_similarity.apply(eval).to_list())
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42, n_init='auto')
kmeans.fit(matrix)
labels = kmeans.labels_
df["Cluster"] = labels 
print("Grouping by clusters")
df.groupby("Cluster").Score.mean().sort_values()

# tsne = TSNE(n_components=2, perplexity=15, random_state=11, init='random', learning_rate=200) 
# vis_dims = tsne.fit_transform(matrix)
# vis_dims.shape
# x = [x for x,y in vis_dims]
# y = [y for x,y in vis_dims]

# fig, ax = plt.subplots(figsize=(10, 7))
# for category, color in enumerate(["purple", "green", "red", "blue", "black", "orange", "brown"]):
#     xs = np.array(x)[df.Cluster == category]
#     ys = np.array(y)[df.Cluster == category]
#     ax.scatter(xs, ys, color=color, alpha=0.3)
#     avg_x = xs.mean()
#     avg_y = ys.mean()
#     ax.scatter(avg_x, avg_y, marker="x", color=color, s=100)

# ax.set_title("Clusters of Fine Food Reviews visualized 2d with K-means", fontsize=14)
# plt.show()

print("Describing clusters...")
rev_per_cluster = 3
for i in range(n_clusters):
    print(f"Cluster {i} Theme:", end=" ")

    reviews = "\n".join(
        df[df.Cluster == i]
        .combined.str.replace("Title: ", "")
        .str.replace("\n\nContent: ", ":  ")
        .sample(rev_per_cluster, random_state=42)
        .values
    )
    print(f"  Cluster {i}")
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f'What do the following customer reviews have in common?\n\nCustomer reviews:\n"""\n{reviews}\n"""\n\nTheme:',
        temperature=0,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    print(response["choices"][0]["text"].replace("\n", ""))

    sample_cluster_rows = df[df.Cluster == i].sample(rev_per_cluster, random_state=42)
    for j in range(rev_per_cluster):
        print(sample_cluster_rows.Score.values[j], end=", ")
        print(sample_cluster_rows.Summary.values[j], end=":   ")
        print(sample_cluster_rows.Text.str[:70].values[j])

    print("-" * 100)

