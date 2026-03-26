from xResNet_50 import xResNet50
import torch
from data_loader import load_processed_mat
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
encoder = xResNet50()
encoder.load_state_dict(torch.load("ecg_encoder_xresnet50_best.pth"))
encoder.eval().cuda()

signals, labels = load_processed_mat("./processed_traindata.mat", labeled=True)
signals = signals.astype(np.float32)
tensor_data = torch.tensor(signals).unsqueeze(1)
dataset = TensorDataset(tensor_data)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

all_features = []
with torch.no_grad():
    for batch in loader:
        x = batch[0].cuda()
        feats = encoder(x)
        all_features.append(feats.cpu())
features = torch.cat(all_features).numpy()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

mask = labels != -1
features_labeled = features[mask]
labels_labeled = labels[mask]

features_2d = TSNE(n_components=2, perplexity=30, init='pca', random_state=42).fit_transform(features_labeled)

plt.figure(figsize=(8, 6))
for label, color, name in zip([0, 1], ['blue', 'red'], ['Non-AF', 'AF']):
    idx = labels_labeled == label
    plt.scatter(features_2d[idx, 0], features_2d[idx, 1], c=color, label=name, alpha=0.6)
plt.legend()
plt.title('ECG Feature Visualization by t-SNE')
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(True)
plt.tight_layout()
plt.show()