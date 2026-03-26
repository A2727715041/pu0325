%% ECG信号处理系统
% 参数初始化
fs = 400;               
target_row = 501;       
ecg = traindata(target_row, :); 
t = (0:length(ecg)-1)/fs;       

% 绘制时域图
figure;
subplot(2, 1, 1);
plot(t, ecg, 'Color', [0 0.6 0.8], 'LineWidth', 1.2);
title('原始ECG信号时域图', 'FontWeight', 'bold');
xlabel('时间 (s)'), ylabel('幅值 (mV)');
grid on;

% 计算频谱（修正关键错误）
nfft = 2^nextpow2(length(ecg));    % 使用2的幂次提高计算效率
window = hamming(length(ecg));     % 汉明窗抑制频谱泄漏
y = fft(ecg.*window, nfft);       % 加窗FFT
P2 = abs(y/nfft);                 
P1 = P2(1:nfft/2+1);              % 单边频谱
P1(2:end-1) = 2*P1(2:end-1);      % 幅值修正（排除Nyquist点）
f = fs*(0:nfft/2)/nfft;           % 正确的频率轴

% 绘制频域图（对数坐标）
subplot(2, 1, 2);
semilogy(f, P1, 'Color', [1 0.4 0], 'LineWidth', 1.2);
title('原始ECG信号频谱', 'FontWeight', 'bold');
xlabel('频率 (Hz)'), ylabel('功率谱密度');
xlim([0 100]);                    % 聚焦关键频段(0-100Hz)
grid on;

