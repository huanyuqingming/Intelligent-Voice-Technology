import os
import numpy as np
import vad_utils
import evaluate
from scipy.io import wavfile
from itertools import groupby

# 分帧
def frame_signal(signal, frame_size, frame_shift, sample_rate):
    # 计算每个帧中的样本数和步进的样本数
    frame_length, frame_step = frame_size * sample_rate, frame_shift * sample_rate  
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))

    # 计算总帧数
    num_frames = int(np.ceil(float(np.abs(len(signal) - frame_length)) / frame_step)) 

    # 补零
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - len(signal)))
    pad_signal = np.append(signal, z) 

    # 分帧
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

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
    i = 1
    for file_name in files:
        base_name = os.path.basename(file_name)
        base_name = base_name[:-4]
        framerate, wave_data = wavfile.read(file_name)
        # 分帧
        frame_size = 0.032
        frame_shift = 0.008
        frames = frame_signal(wave_data, frame_size, frame_shift, framerate)
        # 每个frames是一个文件的所有帧，每个帧是一个512数组
        if labels:
            print("Processing development file", i, "......")
            for j in range(len(frames) - len(labels[base_name])):
                labels[base_name].append(0)   # 补零
            files_labels.extend(labels[base_name])
        files_frames.extend(frames)     # files_frames是所有文件的所有帧，长度为所有帧的总数
        i += 1

    # 提取特征
    energies = short_time_energy(files_frames)
    zcrs = zero_crossing_rate(files_frames)

    return energies, zcrs, files_labels

# 优化预测结果
def opt(labels, smooth_length=10, front_stretched_length=3, back_stretched_length=3):
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
devs = [os.path.join(r'voice-activity-detection-sjtu-spring-2024\vad\wavs\dev', f) for f in os.listdir(r'voice-activity-detection-sjtu-spring-2024\vad\wavs\dev') if f.endswith('.wav')]
tests = [os.path.join(r'voice-activity-detection-sjtu-spring-2024\vad\wavs\test', f) for f in os.listdir(r'voice-activity-detection-sjtu-spring-2024\vad\wavs\test') if f.endswith('.wav')]
# 读取训练集标签
dev_labels = vad_utils.read_label_from_file(r'voice-activity-detection-sjtu-spring-2024\vad\data\dev_label.txt')

# 提取特征
predictions = []
threshold = 60000
print("Extracting features......")
energies, zcrs, files_labels = extract_features(devs, dev_labels)
for i, energy in enumerate(energies):
    if energy > threshold:
        predictions.append(1)
    else:
        predictions.append(0)
print("Features extracted.")

# 测试集预测
threshold = 60000
j = 1
print("Extracting test features......")
with open(r'D:\Intelligent Voice Technology\Project1\task1\test_label.txt', 'w') as f:
    for test in tests:
        print("Processing test file", j, "......")
        j += 1
        test_predictions = []
        a_test = [test]
        test_name = os.path.basename(test)
        test_name = test_name[:-4]
        test_energies, test_zcrs, _ = extract_features(a_test)
        for i, energy in enumerate(test_energies):
            if energy > threshold:
                test_predictions.append(1)
            else:
                test_predictions.append(0)
        smoothed_test_predictions = opt(test_predictions)
        test_labels = vad_utils.prediction_to_vad_label(smoothed_test_predictions)
        f.write(str(test_name) + ' ' + str(test_labels) + '\n')
print("Test features extracted.")

# 开发集评估
print("\nThreshold classifier:")
auc, eer = evaluate.get_metrics(predictions, files_labels)
print("auc:", auc)
print("eer:", eer, "\n")

print("Optimized threshold classifier:")
smoothed_predictions = opt(predictions)
auc, eer = evaluate.get_metrics(smoothed_predictions, files_labels)
print("auc:", auc)
print("eer:", eer)