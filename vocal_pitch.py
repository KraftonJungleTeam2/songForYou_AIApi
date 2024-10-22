import librosa
import crepe
import numpy as np
import pandas as pd
import os
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import butter, filtfilt

def pitch_extract(audio_path, step_size=50) -> None | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    crepe를 이용해 피치 추출

    Args:
    audio_path (str): 폴더

    Returns:
    None | tuple: 성공 시 np.ndarray로 (시간, 주파수, 주파수별 confidence, 전체 confidence 정보 반환)
    """    
    try:
        audio, sr = librosa.load(os.path.join(audio_path, "vocals_preprocessed.wav"))
        # Crepe로 음정 추출
        time, frequency, confidence, activation = crepe.predict(audio, sr, viterbi=False, step_size=step_size)
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

def distorted_gaussian_filter(activation: np.ndarray,
                              degree: int = 2,
                              sigma: int = 3) -> np.ndarray:
    """
    시간축 방향으로 수축한 가우시안 필터를 이용해서 블러처리된 활성화 배열에서 새로운 confidence를 추출

    Args:
        activation (np.ndarray): activation 배열
        degree (int, optional): 수축 정도. 1이면 수축 없음. Defaults to 2.
        sigma (int, optional): 가우시안 필터의 표준편차. 주파수 축 기준.

    Returns:
        np.ndarray: confidence 1차원 배열
    """    
    repeated = np.repeat(activation, degree, axis=0)
    filtered = gaussian_filter(repeated, sigma=3).reshape(-1, degree*360)
    new_confidence = np.max(filtered, axis=1)

    return new_confidence

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

if __name__ == '__main__':
    import matplotlib
    from matplotlib import pyplot as plt
    import numpy as np
    matplotlib.use('TkAgg')

    # result = pitch_extract("results/signal/")
    # if result:
    #     time, frequency, confidence, activation = result
    time = np.load("results/signal/time.npy")
    frequency = np.load("results/signal/frequency.npy")
    confidence = np.load("results/signal/confidence.npy")
    activation = np.load("results/signal/activation.npy")


    squared = activation
    squared = np.repeat(activation, 2, axis=0)
    filtered = gaussian_filter(squared, sigma=3)
    filtered = filtered.reshape(-1, 2*360)
    new_confidence = np.max(filtered, axis=1)
    plt.figure(figsize=(12, 6))
    plt.plot(np.where(confidence>0.5, frequency, np.nan))
    plt.plot(np.where(new_confidence>0.25, frequency, np.nan))
    plt.yscale('log')
    plt.show()
