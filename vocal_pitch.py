import librosa
import crepe
import numpy as np
import pandas as pd
import os
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt

def pitch_extract(audio_path):
    try:
        audio, sr = librosa.load(os.path.join(audio_path, "vocals_preprocessed.wav"))
        # Crepe로 음정 추출
        time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=False, step_size=50)
        frequency = refine(frequency, confidence, 0.5)
        
        np.save(os.path.join(audio_path, "time.npy"), time)
        np.save(os.path.join(audio_path, "frequency.npy"), frequency)
        np.save(os.path.join(audio_path, "confidence.npy"), confidence)
        np.save(os.path.join(audio_path, "activation.npy"), activation)

        return time, frequency, confidence, activation
    except Exception as e:
        print(f"error whlie extracting pitch: {e}")

def refine(frequency, confidence, threshold):
    frequency = np.where(confidence > threshold, frequency, np.nan)
    # smoothed_frequency = smooth_pitch(frequency, window_length=10, polyorder=1)

    frequency = np.where((frequency >= 40)| (frequency <= 2000), frequency, np.nan)

    # 보간 적용
    # smoothed_frequency = interpolate_nearby(frequency, max_gap=20)
    return frequency

def reduce_columns_by_average(arr: np.ndarray, group_size=5) -> np.ndarray:
    """
    이차원 배열에서 각 열을 5개씩 묶어 평균값으로 줄이는 함수.

    Parameters:
        arr (2D numpy array): 입력 이차원 배열
        group_size (int): 묶을 그룹 크기 (기본값: 5)

    Returns:
        (2D numpy array): 열이 1/5로 줄어든 이차원 배열
    """
    rows, cols = arr.shape
    # 새로운 행 크기 계산
    new_rows = rows // group_size
    result = np.zeros((new_rows, cols))

    for j in range(cols):
        for i in range(new_rows):
            # 5개씩 묶어서 평균값 계산
            result[i, j] = np.mean(arr[i*group_size:(i+1)*group_size, j])
    
    return result

# 로우패스 필터 적용
def lowpass_filter(data, cutoff_freq, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def smooth_pitch(pitch, window_length=6, polyorder=3):
    # Savitzky-Golay 필터를 사용하여 스무딩 적용
    # return savgol_filter(pitch, window_length, polyorder)
    return gaussian_filter1d(pitch, sigma=2)

# 보간 함수 정의
def interpolate_nearby(data, max_gap=1):
    # Pandas Series로 변환하여 보간 적용
    series = pd.Series(data)
    result = series.interpolate(method='polynomial', limit=max_gap, limit_direction='both', order=2)

    # 가까운 값이 아닌 경우에는 NaN으로 유지하도록 조건 추가
    result = np.where(result.isna(), np.nan, result)

    return result