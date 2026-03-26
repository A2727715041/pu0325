@echo off
REM SSL+PU学习训练启动脚本

echo === SSL+PU学习训练启动 ===
echo.

echo 正在启动训练流程...
python ecg_ssl_pu/train_ssl_pu_mat.py

if %errorlevel% equ 0 (
    echo.
    echo ✓ 训练完成！
    echo 请查看训练日志和保存的模型文件。
) else (
    echo.
    echo ❌ 训练失败，请检查错误信息。
)

echo.
echo 按任意键退出...
pause >nul