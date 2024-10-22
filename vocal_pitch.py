import librosa
import crepe
import numpy as np
import pandas as pd
import os
from scipy.ndimage import gaussian_filter

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
        frequency = refine(frequency, activation)
        
        np.save(os.path.join(audio_path, "time.npy"), time)
        np.save(os.path.join(audio_path, "frequency.npy"), frequency)
        np.save(os.path.join(audio_path, "confidence.npy"), confidence)
        np.save(os.path.join(audio_path, "activation.npy"), activation)

        return time, frequency, confidence, activation
    except Exception as e:
        print(f"error whlie extracting pitch: {e}")

def refine(frequency, activation, threshold=0.25):
    frequency = np.where((frequency >= 40)| (frequency <= 2000), frequency, np.nan)
    new_confidence = distorted_gaussian_filter(activation, 7)
    new_frequency = confidence_filter(smooth(frequency), new_confidence, threshold)
    
    return new_frequency

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
    filtered = gaussian_filter(repeated, sigma=sigma).reshape(-1, degree*360)
    new_confidence = np.max(filtered, axis=1)

    return new_confidence

# confidence 통과 후 점 제거
def confidence_filter(frequency, confidence, threshold) -> np.ndarray:
    """_summary_
    confidence 통과 후 점을 제거한 frequency를 반환
    Args:
        frequency (_type_): _description_
        confidence (_type_): _description_
        threshold (_type_): _description_

    Returns:
        np.ndarray: _description_
    """    
    filt = np.where(confidence>threshold, frequency, np.nan)
    s = pd.Series(filt)
    mask = s.shift(-1).isna() & s.shift(1).isna()
    s[mask] = np.nan
    result = s.to_numpy()
    
    return result

def smooth(frequency: np.ndarray, degree: float=0.7) -> np.ndarray:
    # 혼자인 점 제거
    # 1옥타브 이상 차이나는 선 제거
    s = pd.Series(frequency)
    mask = s.shift(-1).notna() & s.shift(1).notna()
    interpolated = s.interpolate().where(mask, s).to_numpy()
    result = interpolated
    last = result[0]
    for i in range(1, result.shape[0]-1):
        if np.isnan(result[i]):
            last = np.nan
        else:
            if np.isnan(last):
                last = result[i]
            else:
                if np.abs(np.log2(last/result[i])) < 0.5:
                    last = last*(1-degree) + result[i]*degree
                    result[i] = last
                else:
                    last = np.nan
                    result[i] = last

    return result

if __name__ == '__main__':
    import matplotlib
    from matplotlib import pyplot as plt
    import numpy as np
    matplotlib.use('TkAgg')

    result = pitch_extract("results/signal/")
    if result:
        time, frequency, confidence, activation = result
    # time = np.load("results/signal/time.npy")
    # frequency = np.load("results/signal/frequency.npy")
    # confidence = np.load("results/signal/confidence.npy")
    # activation = np.load("results/signal/activation.npy")

    plt.figure(figsize=(12, 6))
    plt.plot(frequency)
    plt.yscale('log')
    plt.show()