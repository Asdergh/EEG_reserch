import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from matplotlib.colors import ListedColormap as lcm
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import (
    Activation,
    Normalization
)
from tensorflow.keras import Sequential



MIN_WORDS_N = 10
tokenizer = Tokenizer()
norm = Sequential([
    Activation("sigmoid"),
    Normalization()
])

data_path = "C:\\Users\\1\\Downloads\\archive (27)"
train = os.path.join(data_path, "train (2).csv")
with open(train, "r", encoding="utf-8") as f_txt:

    train_buffer = []
    labels = []
    data = f_txt.readlines()

    for string in data[1:]:
        if string.count(";") == 3:

            _, _, text, label = string.split(";")
            tokens_list = text.split(" ")
            if len(tokens_list) > MIN_WORDS_N:

                train_buffer.append(tokens_list[:MIN_WORDS_N])
                labels.append(int(label))



labels = np.asarray(labels)
print(np.unique)

tokenizer.fit_on_texts(train_buffer)
tokens_buffer = np.asarray(tokenizer.texts_to_sequences(train_buffer)).astype(np.float64)
tokens_norm_tensor = norm(tokens_buffer)


print(tokens_norm_tensor, labels)

tree = DecisionTreeClassifier()
pca_dec = PCA(n_components=2)

# rel_features = pca_dec.fit_transform(tokens_norm_tensor)
random_idx = np.random.randint(0, tokens_norm_tensor.shape[1], 2)
rel_features = np.concatenate([
    np.expand_dims(tokens_norm_tensor[:, random_idx[0]], axis=1), 
    np.expand_dims(tokens_norm_tensor[:, random_idx[1]], axis=1)
], axis=1)
rel_features = rel_features[:100]
labels = labels[:100]
print(rel_features.shape)

res = 0.02
X1, X2 = np.meshgrid(
    np.arange(np.min(rel_features[:, 0]), np.max(rel_features[:, 0]), res),
    np.arange(np.min(rel_features[:, 1]), np.max(rel_features[:, 1]), res)
)
tree = tree.fit(rel_features, labels)
pred_logits = tree.predict(
    np.concatenate([np.expand_dims(X1.ravel(), axis=1), np.expand_dims(X2.ravel(), axis=1)], axis=1)
).reshape(X1.shape)

print(X1.shape, pred_logits.shape)


plt.style.use("dark_background")
_, axis = plt.subplots()
own_cmap = lcm(["orange", "gray"])

for i, cll_label in enumerate(np.unique(labels)):
    axis.scatter(rel_features[labels == cll_label, 0], rel_features[labels == cll_label, 1], c=own_cmap(i), s=0.12)

axis.contourf(X1, X2, pred_logits, cmap=own_cmap, alpha=0.26)
plt.show()







