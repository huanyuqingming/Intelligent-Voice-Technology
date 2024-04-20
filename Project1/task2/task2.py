import os
import numpy as np
from tqdm import tqdm
import vad_utils
import evaluate
from scipy.io import wavfile
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from python_speech_features import mfcc
import matplotlib.pyplot as plt

# 定义DNN模型
class dnn(nn.Module):
    def __init__(self):
        super(dnn, self).__init__()
        self.input = nn.Linear(13, 64)
        self.layer = nn.Linear(64, 64)
        self.output = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.input(x))
        x = torch.relu(self.layer(x))
        x = self.sigmoid(self.output(x))
        return x

# 提取特征    
def extract_features(files, labels = {}):
    files_MFCC = []
    files_labels = []
    if labels:
        for file_name in tqdm(files):
            # 提取音频文件
            base_name = os.path.basename(file_name)
            base_name = base_name[:-4]
            sample_rate, signal = wavfile.read(file_name)
            # 提取MFCC特征
            Melfcc = mfcc(signal=signal, samplerate=sample_rate, winlen=0.032, winstep=0.008, nfilt=40, nfft=512, winfunc=np.hamming)
            files_MFCC.extend(Melfcc)
            # 提取标签
            for j in range(len(Melfcc) - len(labels[base_name])):
                labels[base_name].append(0)
            files_labels.extend(labels[base_name])
        # 将列表转换为NumPy数组
        files_MFCC = np.array(files_MFCC)
        files_labels = np.array(files_labels)

        return files_MFCC, files_labels
    else:
        # 提取音频文件
        base_name = os.path.basename(files)
        base_name = base_name[:-4]
        sample_rate, signal = wavfile.read(files)
        # 提取MFCC特征
        Melfcc = mfcc(signal=signal, samplerate=sample_rate, winlen=0.032, winstep=0.008, nfilt=40, nfft=512, winfunc=np.hamming)
        files_MFCC.extend(Melfcc)
        # 将列表转换为NumPy数组
        files_MFCC = np.array(files_MFCC)

        return files_MFCC
    
# 卷积平滑优化
def convolve(outputs, smoothed_length=23):
    smoothed_predictions = np.convolve(outputs.flatten(), np.ones(smoothed_length)/smoothed_length, mode='same') 
    return smoothed_predictions

# 二值化
def binarize(outputs, threshold=0.55):
    outputs = np.where(outputs > threshold, 1, 0)
    return outputs

# # 导入训练集、开发集和测试集
# trains = [os.path.join(r'D:\Project1\voice-activity-detection-sjtu-spring-2024\vad\wavs\train', f) for f in os.listdir(r'D:\Project1\voice-activity-detection-sjtu-spring-2024\vad\wavs\train') if f.endswith('.wav')]
# devs = [os.path.join(r'D:\Project1\voice-activity-detection-sjtu-spring-2024\vad\wavs\dev', f) for f in os.listdir(r'D:\Project1\voice-activity-detection-sjtu-spring-2024\vad\wavs\dev') if f.endswith('.wav')]
# tests = [os.path.join(r'D:\Project1\voice-activity-detection-sjtu-spring-2024\vad\wavs\test', f) for f in os.listdir(r'D:\Project1\voice-activity-detection-sjtu-spring-2024\vad\wavs\test') if f.endswith('.wav')]

# # 读取测试集、开发集标签
# train_labels_dic = vad_utils.read_label_from_file(r'D:\Project1\voice-activity-detection-sjtu-spring-2024\vad\data\train_label.txt')
# dev_labels_dic= vad_utils.read_label_from_file(r'D:\Project1\voice-activity-detection-sjtu-spring-2024\vad\data\dev_label.txt')

# # 提取训练集特征与标签
# print("Extracting training features......")
# train_MFCC, train_labels = extract_features(trains, train_labels_dic)
# print("Training features extracted.\n")

# # 提取开发集特征与标签
# print("Extracting development features......")
# dev_MFCC, dev_labels = extract_features(devs, dev_labels_dic)
# print("Development features extracted.\n")

# # 保存特征和标签
# np.save(r'D:\Project1\task2\train_features.npy', train_features)
# np.save(r'D:\Project1\task2\train_MFCC.npy', train_MFCC)
# np.save(r'D:\Project1\task2\train_labels.npy', train_labels)
# np.save(r'D:\Project1\task2\dev_features.npy', dev_features)
# np.save(r'D:\Project1\task2\dev_MFCC.npy', dev_MFCC)
# np.save(r'D:\Project1\task2\dev_labels.npy', dev_labels)

# 导入特征和标签
print("Loading features......")
train_features = np.load(r'D:\Project1\task2\features\train_features.npy')
train_MFCC = np.load(r'D:\Project1\task2\features\train_MFCC.npy')
train_labels = np.load(r'D:\Project1\task2\features\train_labels.npy')
dev_features = np.load(r'D:\Project1\task2\features\dev_features.npy')
dev_MFCC = np.load(r'D:\Project1\task2\features\dev_MFCC.npy')
dev_labels = np.load(r'D:\Project1\task2\features\dev_labels.npy')
print("Features loaded.\n")



# 创建DNN模型
model = dnn()

# 尝试将模型加载到cuda上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# # 定义损失函数和优化器
# criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

# 将numpy数组转换为torch张量
train_MFCC = torch.from_numpy(train_MFCC).float()
train_labels = torch.from_numpy(train_labels).float().unsqueeze(1)
dev_MFCC = torch.from_numpy(dev_MFCC).float()
dev_labels = torch.from_numpy(dev_labels).float().unsqueeze(1)

# # 创建数据加载器
# train_dataset = TensorDataset(train_MFCC, train_labels)
# train_dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=True)

# # 训练模型
# train_losses = []
# dev_losses = []

# for epoch in range(100):
#     print("Training......")
#     model.train()
#     loss = 0
#     for features, labels in tqdm(train_dataloader):
#         features = features.to(device)
#         labels = labels.to(device)
#         # 前向传播
#         outputs = model(features)
#         a_loss = criterion(outputs, labels)
#         # 反向传播和优化
#         optimizer.zero_grad()
#         a_loss.backward()
#         optimizer.step()
#         loss += a_loss.item()
#     torch.save(model.state_dict(), f'D:\\Project1\\task2\\model\\model+{epoch+1}epoch.pth')
#     loss /= len(train_dataloader)
#     train_losses.append(loss)
#     print("Training finished.", f'Epoch {epoch+1}, Training Loss: {loss}')

#     # 测试模型
#     print("Testing......")
#     model.eval()
#     dev_MFCC = dev_MFCC.to(device)
#     dev_labels = dev_labels.to(device)
#     outputs = model(dev_MFCC)
#     loss = criterion(outputs, dev_labels)
#     dev_losses.append(loss.item())
#     outputs = outputs.detach().cpu().numpy()
#     dev_labels = dev_labels.cpu()
#     auc, eer, acc, thres = evaluate.get_metrics(outputs, dev_labels, 0.55, plot=False)

#     print("auc:", auc, "eer:", eer, "acc:", acc, "thres:", thres, "test_loss:", loss.item())

#     print("Testing finished.\n")

# # 绘制损失曲线
# plt.plot(train_losses, label='Training Loss')
# plt.plot(dev_losses, label='Development Loss')
# plt.title('Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

# 导入模型
model.load_state_dict(torch.load(f'D:\\Project1\\task2\\model\\MFCC\\model+100epoch.pth'))

# 开发集评估
threshold = 0.55
model.eval()
dev_MFCC = dev_MFCC.to(device)
dev_labels = dev_labels.to(device) 
outputs = model(dev_MFCC)
outputs = outputs.detach().cpu().numpy()
dev_labels = dev_labels.cpu()
# 卷积平滑
smoothed_predictions = convolve(outputs)
auc, eer, acc = evaluate.get_metrics(smoothed_predictions, dev_labels, threshold, plot=False)
print("Optimized DNN classifier:")
print("auc:", auc, "eer:", eer, "acc:", acc)

# # 测试集预测
# print("Predicting test set......")
# with open(r'D:\Project1\task2\test_label.txt', 'w') as f:
#     for test in tqdm(tests):
#         test_name = os.path.basename(test)
#         test_name = test_name[:-4]
#         test_features = torch.from_numpy(extract_features(test)).to(device).float()
#         outputs = model(test_features)
#         outputs = outputs.cpu().detach().numpy()
#         smoothed_test_predictions = convolve(outputs)
#         labels = binarize(smoothed_test_predictions)
#         test_labels = vad_utils.prediction_to_vad_label(smoothed_test_predictions)
#         f.write(str(test_name) + ' ' + str(test_labels) + '\n')
# print("Test set predicted.\n")
