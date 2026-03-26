from xResNet_50 import xResNet50
import torch
from data_loader import load_processed_mat
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

encoder = xResNet50()
encoder.load_state_dict(torch.load("ecg_encoder_xresnet50_best.pth"))
encoder.eval().cuda()

signals, labels = load_processed_mat("D:/CPSC2025/processed_traindata.mat")
tensor_data = torch.tensor(signals[:1000]).unsqueeze(1)
dataset = TensorDataset(tensor_data)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

features = []
with torch.no_grad():
    for batch in loader:
        x = batch[0].cuda().float()
        feat = encoder(x)
        features.append(feat.cpu())
features = torch.cat(features).numpy()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(features, labels[:1000], test_size=0.2, stratify=labels[:1000], random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
sen = tp / (tp + fn + 1e-8)
spe = tn / (tn + fp + 1e-8)
macc = (sen + spe) / 2

print(f"Accuracy: {acc:.4f}")
print(f"Sensitivity: {sen:.4f}")
print(f"Specificity: {spe:.4f}")
print(f"Macc (avg): {macc:.4f}")