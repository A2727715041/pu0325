# === 优化后的 main.py，无 checkpoint，启用 Early Stopping + Macc 验证 ===
import torch
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np

from xResNet_50 import xResNet50
from Online_and_Target import ECG_SRL
from Augmentation import ECGMatDataset, ecg_augment_pair
from data_loader import load_processed_mat
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# 初始化模型与优化器
encoder = xResNet50()
model = ECG_SRL(encoder).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 数据加载
# train_dataset = ECGMatDataset("./processed_traindata.mat")
# 修改后：加载1000到20000的索引范围（含1000，不含20000，即1000~19999）
train_dataset = ECGMatDataset(
    file_path="./processed_traindata.mat",
    start_idx=1000,
    end_idx=20000,
    labeled=False  # 训练集无需标签
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)

# 设置
num_epochs = 50
patience = 20
patience_counter = 0
best_macc = 0.0
validate_every = 1
best_model_path = "ecg_encoder_xresnet50_best.pth"

# 评估函数（用前1000条数据）
def evaluate_encoder(encoder):
    signals, labels = load_processed_mat("./processed_traindata.mat")
    signals = signals[:1000].astype(np.float32)
    labels = labels[:1000]

    with torch.no_grad():
        feats = []
        dataset = TensorDataset(torch.tensor(signals).unsqueeze(1))
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        for batch in loader:
            x = batch[0].cuda().float()
            feat = encoder(x)
            feats.append(feat.cpu())
        features = torch.cat(feats).numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sen = tp / (tp + fn + 1e-8)
    spe = tn / (tn + fp + 1e-8)
    macc = (sen + spe) / 2
    return macc

# 正式训练
total_start = time.time()
try:
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()

        for batch in train_loader:
            x_np = batch.numpy()
            x1_list, x2_list = [], []
            for sig in x_np:
                x1, x2 = ecg_augment_pair(sig[0])
                x1_list.append(x1)
                x2_list.append(x2)
            x1 = torch.stack(x1_list).cuda().float()
            x2 = torch.stack(x2_list).cuda().float()

            loss = model(x1, x2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_time = time.time() - epoch_start
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss / len(train_loader):.4f} | Time: {epoch_time:.2f}s")

        # 验证阶段
        if (epoch + 1) % validate_every == 0:
            model.eval()
            macc = evaluate_encoder(model.encoder_q)
            print(f"--> 验证阶段：Macc = {macc:.4f}")

            if macc > best_macc:
                print(f"新的最优 Macc: {macc:.4f}（之前为 {best_macc:.4f}），保存模型")
                best_macc = macc
                patience_counter = 0
                torch.save(model.encoder_q.state_dict(), best_model_path)
            else:
                patience_counter += 1
                print(f"Macc 未提升，patience_counter = {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("提前停止训练：Macc 多轮未提升")
                    break

except KeyboardInterrupt:
    print("手动中断训练，训练提前结束。")

total_time = (time.time() - total_start) / 60
print(f"\n总训练耗时：{total_time:.2f} 分钟")
