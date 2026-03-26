# 导入PyTorch库，用于构建和训练神经网络
import torch
# 导入PyTorch的神经网络模块，包含各种层和激活函数等
import torch.nn as nn
# 导入PyTorch的函数式接口，包含各种操作函数
import torch.nn.functional as F

# 定义残差块类，继承自nn.Module（PyTorch中所有神经网络模块的基类）
class ResidualBlock(nn.Module):
    # 初始化方法，接收输入通道数、输出通道数和步长（默认为1）
    def __init__(self, in_channels, out_channels, stride=1):
        # 调用父类nn.Module的初始化方法
        super().__init__()
        # 定义第一个卷积层：1D卷积，输入通道in_channels，输出通道out_channels，卷积核大小3，步长stride，填充1，不使用偏置
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # 定义第一个批归一化层：对输出通道数out_channels进行批归一化
        self.bn1 = nn.BatchNorm1d(out_channels)
        # 定义ReLU激活函数，inplace=True表示直接在原张量上修改，节省内存
        self.relu = nn.ReLU(inplace=True)
        # 定义第二个卷积层：1D卷积，输入输出通道均为out_channels，卷积核大小3，填充1，不使用偏置
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        # 定义第二个批归一化层：对输出通道数out_channels进行批归一化
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 初始化下采样模块为None
        self.downsample = None
        # 当步长不等于1或输入输出通道数不同时，需要下采样使维度匹配
        if stride != 1 or in_channels != out_channels:
            # 下采样模块：1x1卷积（调整通道数）+ 批归一化（保持分布）
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    # 前向传播方法，定义数据在模块中的流动过程
    def forward(self, x):
        # 保存输入x作为跳跃连接的恒等映射
        identity = x
        # 第一个卷积层 -> 批归一化 -> ReLU激活
        out = self.relu(self.bn1(self.conv1(x)))
        # 第二个卷积层 -> 批归一化（此时不激活，等待与恒等映射相加后再激活）
        out = self.bn2(self.conv2(out))
        # 如果存在下采样模块，对恒等映射进行下采样以匹配维度
        if self.downsample:
            identity = self.downsample(x)
        # 将卷积后的输出与恒等映射相加（残差连接）
        out += identity
        # 对相加结果应用ReLU激活并返回
        return self.relu(out)

# 定义xResNet50模型类，继承自nn.Module
class xResNet50(nn.Module):
    # 初始化方法，定义模型的各个组件
    def __init__(self):
        # 调用父类nn.Module的初始化方法
        super().__init__()
        # 定义stem（茎干）部分：一系列卷积和激活，用于初步特征提取
        self.stem = nn.Sequential(
            # 1D卷积：输入通道1（可能是单通道序列数据），输出32，核大小3，步长2，填充1，无偏置
            nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32),  # 批归一化
            nn.ReLU(inplace=True),  # ReLU激活
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),  # 保持通道数32，步长1
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),  # 通道数从32升到64
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        # 定义stage1：由3个残差块组成，输入输出通道64，步长1（不改变尺寸）
        self.stage1 = self._make_layer(64, 64, 3, stride=1)
        # 定义stage2：由4个残差块组成，输入64→输出128，步长2（尺寸减半）
        self.stage2 = self._make_layer(64, 128, 4, stride=2)
        # 定义stage3：由6个残差块组成，输入128→输出256，步长2（尺寸减半）
        self.stage3 = self._make_layer(128, 256, 6, stride=2)
        # 定义stage4：由3个残差块组成，输入256→输出512，步长2（尺寸减半）
        self.stage4 = self._make_layer(256, 512, 3, stride=2)
        # 全局平均池化：将每个通道的序列长度压缩为1（适应任意输入长度）
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # 定义特征维度为512（与stage4输出通道一致）
        self.feature_dim = 512

    # 辅助方法：构建由多个残差块组成的stage
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        # 第一个残差块需要调整通道数和步长（可能下采样）
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        # 后续blocks-1个残差块：输入输出通道一致，步长1（不改变尺寸）
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        # 将所有残差块组合成一个序列容器
        return nn.Sequential(*layers)

    # 前向传播方法：定义整个模型的数据流动
    def forward(self, x):
        # 输入x经过stem部分提取初步特征
        x = self.stem(x)
        # 依次经过各个stage进行特征深化
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        # 全局池化：将特征图压缩为[batch_size, 512, 1]
        x = self.global_pool(x)
        # 将特征展平为[batch_size, 512]并返回
        return x.view(x.size(0), -1)