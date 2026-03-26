% 假设你已经加载了数据并且变量名为 traindata
% 如果变量名不同，请替换下面的 'traindata' 为实际变量名

% 加载数据（如果尚未加载）
% load('traindata.mat'); % 根据实际情况取消注释此行

% 确认变量存在
if ~exist('traindata', 'var')
    error('未找到名为 traindata 的变量，请检查你的 .mat 文件内容');
end

% 创建一个新的图形窗口
figure;

% 设置网格布局，这里使用2x5的布局来放置10个子图
for i = 50:60
    figure; % 每次迭代时新开一个图形窗口
    plot(traindata(i, :)); % 绘制数据
    title(['数据 ', num2str(i)]); % 设置标题
    xlabel('特征索引'); % X轴标签
    ylabel('值'); % Y轴标签
end

% 调整子图之间的间距，使布局更美观
sgtitle('前十条数据的可视化'); % 为整个图形窗口添加一个总标题（可选）