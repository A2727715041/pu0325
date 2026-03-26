# 导入必要的PyTorch库
import torch  # 导入PyTorch主库，用于张量操作和深度学习计算
import torch.nn as nn  # 导入神经网络模块，包含各种层和损失函数等
import torch.nn.functional as F  # 导入神经网络功能模块，包含各种激活函数、池化操作等
from copy import deepcopy  # 导入深拷贝函数，用于复制对象及其内部数据

# 定义MLPHead类，作为多层感知机头，用于特征转换
class MLPHead(nn.Module):  # 继承nn.Module，成为PyTorch中的一个神经网络模块
    def __init__(self, in_dim=512, hidden_dim=128, out_dim=128):  # 初始化方法，定义输入、隐藏、输出维度
        super().__init__()  # 调用父类nn.Module的初始化方法
        # 定义一个顺序容器，包含线性层和激活函数
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),  # 线性层：将输入维度转换为隐藏层维度
            nn.ReLU(inplace=True),  # ReLU激活函数，inplace=True表示原地操作节省内存
            nn.Linear(hidden_dim, out_dim)  # 线性层：将隐藏层维度转换为输出维度
        )

    def forward(self, x):  # 前向传播方法，定义数据流向
        return self.net(x)  # 将输入x传入定义好的网络结构并返回结果

# 定义ECG_SRL类，用于心电图的自监督表示学习模型
class ECG_SRL(nn.Module):  # 继承nn.Module，作为神经网络模块
    def __init__(self, encoder, proj_dim=128, momentum=0.99):  # 初始化方法，接收编码器、投影维度和动量参数
        super().__init__()  # 调用父类初始化方法
        self.encoder_q = encoder  # 定义查询编码器（可训练）
        # 定义查询投影头，输入维度为编码器的特征维度，隐藏层128，输出维度proj_dim
        self.projector_q = MLPHead(encoder.feature_dim, 128, proj_dim)
        # 定义查询预测头，输入输出维度均为proj_dim，隐藏层128
        self.predictor_q = MLPHead(proj_dim, 128, proj_dim)

        self.encoder_k = deepcopy(encoder)  # 定义动量编码器（ deepcopy复制编码器，避免共享参数 ）
        self.projector_k = deepcopy(self.projector_q)  # 动量投影头（深拷贝查询投影头）

        # 冻结动量编码器的参数，不参与梯度更新
        for param in self.encoder_k.parameters():
            param.requires_grad = False  # 设为False表示参数不需要计算梯度
        # 冻结动量投影头的参数，不参与梯度更新
        for param in self.projector_k.parameters():
            param.requires_grad = False

        self.momentum = momentum  # 动量参数，用于更新动量编码器和投影头

    @torch.no_grad()  # 装饰器，表示该方法内的操作不计算梯度，节省内存
    def momentum_update(self):  # 动量更新方法，更新动量编码器和投影头的参数
        # 更新动量编码器参数：动量*旧参数 + (1-动量)*查询编码器参数
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data
        # 更新动量投影头参数：同上，基于查询投影头参数更新
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data

    def forward(self, x1, x2):  # 前向传播方法，接收两个增强后的样本x1和x2
        h1 = self.encoder_q(x1)  # x1经过查询编码器得到特征h1
        z1 = self.projector_q(h1)  # h1经过查询投影头得到投影特征z1
        q1 = self.predictor_q(z1)  # z1经过查询预测头得到预测特征q1

        with torch.no_grad():  # 对动量编码器相关操作禁用梯度计算
            h2 = self.encoder_k(x2)  # x2经过动量编码器得到特征h2
            z2 = self.projector_k(h2)  # h2经过动量投影头得到投影特征z2

        q1 = F.normalize(q1, dim=1)  # 对q1在维度1上进行L2归一化
        z2 = F.normalize(z2, dim=1)  # 对z2在维度1上进行L2归一化

        # 计算损失：2 - 2*均值(逐样本q1与z2的点积)，本质是基于余弦相似度的对比损失
        loss = 2 - 2 * (q1 * z2).sum(dim=1).mean()
        self.momentum_update()  # 执行动量更新，更新动量编码器和投影头
        return loss  # 返回计算得到的损失值
