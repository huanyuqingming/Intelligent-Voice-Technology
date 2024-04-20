import os
import numpy as np
from tqdm import tqdm
import vad_utils
import evaluate
from scipy.io import wavfile
from itertools import groupby
import matplotlib.pyplot as plt

# 分帧
def frame_signal(signal, frame_size, frame_shift, sample_rate):
    # 计算帧长和帧移
    frame_length, frame_step = frame_size * sample_rate, frame_shift * sample_rate  
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))

    # 计算总帧数
    num_frames = int(np.ceil(float(np.abs(len(signal) - frame_length)) / frame_step)) + 1 

    # 补零
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - len(signal)))
    pad_signal = np.append(signal, z) 

    # 分帧
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # 创建汉明窗
    hamming = np.hamming(frame_length)

    # 将汉明窗应用到每个帧上
    frames *= hamming

    return frames

# 计算短时能量
def short_time_energy(frames):
    frames = np.array(frames)
    ste = np.sum(frames ** 2, axis=1) / 512
    return ste

# 计算过零率
def zero_crossing_rate(frames):
    zcr = 0.5 * np.mean(np.abs(np.diff(np.greater_equal(frames, 0).astype(int))), axis=1)
    return zcr

# 提取特征
def extract_features(files, labels={}):
    
    files_frames = []
    files_labels = []
    if labels:
        for file_name in tqdm(files):
            base_name = os.path.basename(file_name)
            base_name = base_name[:-4]
            sample_rate, wave_data = wavfile.read(file_name)
            # 分帧
            frame_size = 0.032
            frame_shift = 0.008
            frames = frame_signal(wave_data, frame_size, frame_shift, sample_rate)
            # 每个frames是一个文件的所有帧，每个帧是一个512数组
            for j in range(len(frames) - len(labels[base_name])):
                labels[base_name].append(0)   # 补零
            files_labels.extend(labels[base_name])
            files_frames.extend(frames)     # files_frames是所有文件的所有帧，长度为所有帧的总数
    else:
        for file_name in files:
            base_name = os.path.basename(file_name)
            base_name = base_name[:-4]
            sample_rate, wave_data = wavfile.read(file_name)
            # 分帧
            frame_size = 0.032
            frame_shift = 0.008
            frames = frame_signal(wave_data, frame_size, frame_shift, sample_rate)
            # 每个frames是一个文件的所有帧，每个帧是一个512数组
            files_frames.extend(frames)     # files_frames是所有文件的所有帧，长度为所有帧的总数
    
    # 提取特征
    energies = short_time_energy(files_frames)
    zcrs = zero_crossing_rate(files_frames)
    features = np.array([energies, zcrs]).T

    return features, files_labels, wave_data

# 优化预测结果
def opt(labels, smooth_length=17, front_stretched_length=3, back_stretched_length=3):
    # 平滑音频，减少音频中的短暂静音
    result = []
    for key, group in groupby(labels):
        group = list(group)
        if key == 0 and len(group) < smooth_length:
            result.extend([1] * len(group))
        else:
            result.extend(group)

    # 加入语音前后清音区
    n = len(result) # 前清音区
    for i in range(n):
        if result[i] == 1:
            for j in range(max(0, i - front_stretched_length), i):
                result[j] = 1
    result.reverse()
    for i in range(n):  # 后清音区
        if result[i] == 1:
            for j in range(max(0, i - back_stretched_length), i):
                result[j] = 1
    result.reverse()

    return result

# 导入开发集与测试集
devs = [os.path.join(r'D:\Project1\voice-activity-detection-sjtu-spring-2024\vad\wavs\dev', f) for f in os.listdir(r'D:\Project1\voice-activity-detection-sjtu-spring-2024\vad\wavs\dev') if f.endswith('.wav')]
tests = [os.path.join(r'D:\Project1\voice-activity-detection-sjtu-spring-2024\vad\wavs\test', f) for f in os.listdir(r'D:\Project1\voice-activity-detection-sjtu-spring-2024\vad\wavs\test') if f.endswith('.wav')]
# 读取开发集标签
dev_labels = vad_utils.read_label_from_file(r'D:\Project1\voice-activity-detection-sjtu-spring-2024\vad\data\dev_label.txt')

# 提取特征
predictions = []
threshold = [35000, 0.005]
print("Extracting development features......")
features, files_labels, wave_data = extract_features(devs, dev_labels)
mask = (features[:, 0] > threshold[0]) & (features[:, 1] > threshold[1])
predictions = np.where(mask, 1, 0)
print("Development features extracted.\n")

# # 测试集预测
# print("Extracting test features......")
# with open(r'D:\Project1\task1\test_label.txt', 'w') as f:
#     for test in tqdm(tests):
#         test_predictions = []
#         a_test = [test]
#         test_name = os.path.basename(test)
#         test_name = test_name[:-4]
#         test_features, _, _ = extract_features(a_test)
#         for test_feature in test_features:
#             if test_feature[0] > threshold[0] and test_feature[1] > threshold[1]:
#                 test_predictions.append(1)
#             else:
#                 test_predictions.append(0)
#         smoothed_test_predictions = opt(test_predictions)
#         test_labels = vad_utils.prediction_to_vad_label(smoothed_test_predictions)
#         f.write(str(test_name) + ' ' + str(test_labels) + '\n')
# print("Test features extracted.")

# 开发集评估
print("Optimized threshold classifier:")
smoothed_predictions = opt(predictions)
auc, eer, acc = evaluate.get_metrics(smoothed_predictions, files_labels, plot=True)
print("auc:", auc, "eer:", eer, "acc:", acc)


# # 绘制预测图（在只有一个文件的时候用！！！）
# plt.figure()

# # 绘制语音预测段
# x_wave = np.linspace(0, len(wave_data)/16e3, len(wave_data))
# x_pred = np.linspace(0, len(wave_data)/16e3, len(predictions))
# predictions_interp = np.interp(x_wave, x_pred, predictions)
# smoothed_predictions_interp = np.interp(x_wave, x_pred, smoothed_predictions)
# files_labels_interp = np.interp(x_wave, x_pred, files_labels)

# # 绘制波形图
# plt.plot(x_wave, wave_data, label='Wave data')

# # 绘制语音预测段
# plt.plot(x_wave, 9000*predictions_interp, label='Prediction Label')
# plt.plot(x_wave, 10000*smoothed_predictions_interp, label='Smoothed Prediction Label')
# plt.plot(x_wave, 11000*files_labels_interp, label='Truth Label')


# plt.title('Waveform')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.show()