# 导入numpy库，用于数值计算和数组操作，这里用别名np简化调用
import numpy as np
# 从scipy.io模块导入loadmat函数，用于加载.mat格式的MATLAB数据文件
from scipy.io import loadmat

# 定义一个函数，用于加载处理后的.mat数据文件，参数file_path是文件路径，labeled表示是否需要返回标签（默认True）
def load_processed_mat(file_path, labeled=True):
    # 使用loadmat函数加载指定路径的.mat文件，返回一个字典类型的数据
    data = loadmat(file_path)
    # 根据文件路径中是否包含'train'字符串，确定要提取的数据键名：训练数据用'processed_traindata'，否则（测试数据）用'processed_testdata'
    key_name = 'processed_traindata' if 'train' in file_path else 'processed_testdata'
    # 从加载的data字典中，通过上面确定的key_name提取信号数据（通常是一个二维数组，形状为[样本数, 特征数]）
    signals = data[key_name]
    # === 添加标准化 ===
    # 计算均值和标准差（按样本维度，axis=1表示对每个样本的特征轴标准化）
    mean = np.mean(signals, axis=1, keepdims=True)
    std = np.std(signals, axis=1, keepdims=True)
    # 避免除以0，给std加一个小epsilon
    signals = (signals - mean) / (std + 1e-8)

    # 如果需要返回标签（labeled为True）
    if labeled:
        # 创建一个与信号样本数相同的全0数组，数据类型为整数，用于存储标签
        labels = np.zeros(signals.shape[0], dtype=int)
        # 将前500个样本的标签设为1，表示AF（心房颤动）类别
        labels[:500] = 1  # AF
        # 将第500到999个样本（共500个）的标签设为0，表示非AF类别
        labels[500:1000] = 0  # Non-AF
        # 将第1000个及以后的样本标签设为-1（表示未标注类别）
        labels[1000:] = -1
        # 返回信号数据和对应的标签
        return signals, labels
    else:
        # 如果不需要标签，返回信号数据和None
        return signals, None