#!/bin/bash
# SSL+PU学习训练启动脚本

echo "=== SSL+PU学习训练启动 ==="
echo ""

echo "正在启动训练流程..."
python ecg_ssl_pu/train_ssl_pu_mat.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 训练完成！"
    echo "请查看训练日志和保存的模型文件。"
else
    echo ""
    echo "❌ 训练失败，请检查错误信息。"
fi

echo ""
echo "按任意键退出..."
read -n 1