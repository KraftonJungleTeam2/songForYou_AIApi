import librosa
# import crepe
from crepe.core import model_srate, as_strided, to_viterbi_cents, to_local_average_cents, build_and_load_model
import numpy as np
import pandas as pd
import os
from scipy.ndimage import gaussian_filter
import webrtcvad
import wave
import numpy as np
from pydub import AudioSegment

# https://github.com/marl/crepe
def predict(audio, sr, model_capacity='full',
            viterbi=False, center=True, step_size=10, verbose=1):
    """
    Perform pitch estimation on given audio

    Parameters
    ----------
    audio : np.ndarray [shape=(N,) or (N, C)]
        The audio samples. Multichannel audio will be downmixed.
    sr : int
        Sample rate of the audio samples. The audio will be resampled if
        the sample rate is not 16 kHz, which is expected by the model.
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity; see the docstring of
        :func:`~crepe.core.build_and_load_model`
    viterbi : bool
        Apply viterbi smoothing to the estimated pitch curve. False by default.
    center : boolean
        - If `True` (default), the signal `audio` is padded so that frame
          `D[:, t]` is centered at `audio[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
    step_size : int
        The step size in milliseconds for running pitch estimation.
    verbose : int
        Set the keras verbosity mode: 1 (default) will print out a progress bar
        during prediction, 0 will suppress all non-error printouts.

    Returns
    -------
    A 4-tuple consisting of:

        time: np.ndarray [shape=(T,)]
            The timestamps on which the pitch was estimated
        frequency: np.ndarray [shape=(T,)]
            The predicted pitch values in Hz
        confidence: np.ndarray [shape=(T,)]
            The confidence of voice activity, between 0 and 1
        activation: np.ndarray [shape=(T, 360)]
            The raw activation matrix
    """
    activation = get_activation(audio, sr, model_capacity=model_capacity,
                                center=center, step_size=step_size,
                                verbose=verbose)
    confidence = activation.max(axis=1)

    if viterbi:
        cents = to_viterbi_cents(activation)
    else:
        cents = to_local_average_cents(activation)

    frequency = 10 * 2 ** (cents / 1200)
    # frequency[np.isnan(frequency)] = 0

    time = np.arange(confidence.shape[0]) * step_size / 1000.0

    return time, frequency, confidence, activation
# https://github.com/marl/crepe
def get_activation(audio, sr, model_capacity='full', center=True, step_size=10,
                   verbose=1):
    """

    Parameters
    ----------
    audio : np.ndarray [shape=(N,) or (N, C)]
        The audio samples. Multichannel audio will be downmixed.
    sr : int
        Sample rate of the audio samples. The audio will be resampled if
        the sample rate is not 16 kHz, which is expected by the model.
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity; see the docstring of
        :func:`~crepe.core.build_and_load_model`
    center : boolean
        - If `True` (default), the signal `audio` is padded so that frame
          `D[:, t]` is centered at `audio[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
    step_size : int
        The step size in milliseconds for running pitch estimation.
    verbose : int
        Set the keras verbosity mode: 1 (default) will print out a progress bar
        during prediction, 0 will suppress all non-error printouts.

    Returns
    -------
    activation : np.ndarray [shape=(T, 360)]
        The raw activation matrix
    """
    model = build_and_load_model('full')

    if len(audio.shape) == 2:
        audio = audio.mean(1)  # make mono
    audio = audio.astype(np.float32)
    if sr != model_srate:
        # resample audio if necessary
        from resampy import resample
        audio = resample(audio, sr, model_srate)

    # pad so that frames are centered around their timestamps (i.e. first frame
    # is zero centered).
    if center:
        audio = np.pad(audio, 512, mode='constant', constant_values=0)

    # make 1024-sample frames of the audio with hop length of 10 milliseconds
    hop_length = int(model_srate * step_size / 1000)
    n_frames = 1 + int((len(audio) - 1024) / hop_length)
    frames = as_strided(audio, shape=(1024, n_frames),
                        strides=(audio.itemsize, hop_length * audio.itemsize))
    frames = frames.transpose().copy()

    # normalize each frame -- this is expected by the model
    frames -= np.mean(frames, axis=1)[:, np.newaxis]
    frames /= np.clip(np.std(frames, axis=1)[:, np.newaxis], 1e-8, None)

    # run prediction and convert the frequency bin weights to Hz
    return model.predict(frames, verbose=verbose)

def pitch_extract(audio_path, lyrics: dict={}, step_size=50):
    """
    crepe를 이용해 피치 추출

    Args:
    audio_path (str): 폴더

    Returns:
    None | tuple: 성공 시 np.ndarray로 (시간, 주파수, 주파수별 confidence, 전체 confidence 정보 반환)
    """
    audio, sr = librosa.load(os.path.join(audio_path, "vocals.wav"))
    # Crepe로 음정 추출
    _, frequency, _, activation = predict(audio, sr, viterbi=True, step_size=step_size)
    frequency, confidence = refine(frequency, activation, audio_path)
    
    np.save(os.path.join(audio_path, "frequency.npy"), frequency)
    np.save(os.path.join(audio_path, "confidence.npy"), confidence)
    np.save(os.path.join(audio_path, "activation.npy"), activation)

    return frequency, confidence, activation

def refine(frequency, activation, audio_path, threshold=0.25):
    # frequency 이상치 제거
    Q1 = np.nanpercentile(frequency, 25)
    Q3 = np.nanpercentile(frequency, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    frequency = np.where(frequency <= upper_bound, frequency, np.nan)
    frequency = np.where(frequency >= lower_bound, frequency, np.nan)

    # 가사로 마스킹 => 좋은 생각이 아닌 듯
    no_vocal_masking(activation, audio_path)
    # 가우시안 필터
    new_confidence = distorted_gaussian_filter(activation, 7)
    # 스무딩 & new_frequency 생성
    new_frequency = smooth(confidence_filter(frequency, new_confidence, threshold))
    
    
    return new_frequency, new_confidence

def no_vocal_masking(data: np.ndarray, audio_path):
    audio = AudioSegment.from_file(os.path.join(audio_path, "vocals.wav"))
    
    # Mono로 변환 (VAD는 Mono만 지원)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(48000)  # VAD가 지원하는 샘플 레이트로 설정 (8000, 16000, 32000, 48000 중 하나)

    # WAV 파일로 저장하여 WebRTC VAD 적용
    audio.export(os.path.join(audio_path, "vocals_mono.wav"), format="wav")
    
    # WebRTC VAD 설정
    vad = webrtcvad.Vad()
    vad.set_mode(2)  # 모드 3: 가장 민감한 설정

    # WAV 파일 읽기
    with wave.open(os.path.join(audio_path, "vocals_mono.wav"), 'rb') as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        frame_length = int(sample_rate * 0.03)

    # 음성 데이터를 NumPy 배열로 변환
    samples = np.frombuffer(frames, dtype=np.int16)

    # 음성 구간 탐지
    segments = [0, 0]
    for i in range(0, len(samples), frame_length):
        frame = samples[i:i + frame_length]
        if len(frame) < frame_length:
            break

        # VAD를 이용해 음성인지 여부 확인
        is_speech = vad.is_speech(frame.tobytes(), sample_rate)
        if not is_speech:
            if (segments[-1] == i-frame_length):
                segments[-1] = i
            else:
                segments.append(i)
                segments.append(i)

    for i in range(0, len(segments), 2):
        s, e = segments[i:i+2]
        s = int(s/48/50) # 48000hz 샘플 기준 인덱스를 20hz(개당 50ms) 기준 인덱스로 변환
        e = int(e/48/50)
        data[s:e] = 0

    return data


# 끝부분 치켜 올라가는 부분 제거
def cut_tail(frequency):
    up_streak = 0
    down_streak = 0
    last = 0
    frequency_ = frequency.copy()
    for i, v in enumerate(frequency_):
        if np.isnan(v):
            if up_streak > 0 and (up_streak > 2 or frequency_[i-up_streak-1]/frequency_[i-1] ** (1/up_streak) < 0.75):
                frequency[i-up_streak:i] = np.nan
            if down_streak > 0 and (down_streak > 2 or frequency_[i-down_streak-1]/frequency_[i-1] ** (1/down_streak) > 1.33):
                frequency[i-down_streak-1:i] = np.nan
            up_streak = 0
            down_streak = 0
        elif v > last:
            if down_streak > 0 and (down_streak > 2 or frequency_[i-down_streak-1]/frequency_[i-1] ** (1/down_streak) > 1.33):
                frequency[i-down_streak-1:i] = np.nan
            
            up_streak += 1
            down_streak = -1
        elif v < last:
            if down_streak > -1:
                down_streak += 1
            up_streak = 0
        last = v
    
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
    mask = confidence > threshold
    flexible_mask = confidence > threshold*0.8
    for i in range(1, confidence.shape[0]):
        if mask[i-1] and not mask[i] and flexible_mask[i]:
            mask[i] = True
        
    filt = np.where(mask, frequency, np.nan)
    s = pd.Series(filt)
    mask = s.shift(-1).isna() & s.shift(1).isna()
    s[mask] = np.nan
    result = s.to_numpy()
    
    return result

def smooth(frequency: np.ndarray, degree: float=0.7) -> np.ndarray:
    # 혼자인 점 제거
    # 1옥타브 이상 차이나는 선 제거
    result = frequency.copy()
    last = result[0]
    for i in range(1, result.shape[0]-1):
        # 현재 값이 nan인 경우 다음으로
        if np.isnan(result[i]):
            last = np.nan
        # 이전 값이 nan인 경우 다음으로
        elif np.isnan(last):
            last = result[i]
        # 차이가 0.5옥타브 미만인 경우
        elif np.abs(np.log2(last/result[i])) < 0.5:
            last = last*(1-degree) + result[i]*degree
            result[i] = last
        # 차이가 0.5옥타브 이상인 경우
        else:
            last = np.nan
            result[i] = last
            
    s = pd.Series(result)
    mask = s.shift(-1).notna() & s.shift(1).notna()
    result = s.interpolate().where(mask, s).to_numpy()
    cut_tail(result)

    return result

if __name__ == '__main__':
    import matplotlib
    from matplotlib import pyplot as plt
    import numpy as np
    matplotlib.use('TkAgg')

    audio_path = "."
    audio, sr = librosa.load(os.path.join(audio_path, "vocals_preprocessed.wav"))
    # Crepe로 음정 추출
    _, frequency, confidence, activation = predict(audio, sr, viterbi=True, step_size=25)

    plt.subplot(2, 1, 1)
    plt.plot(np.where(confidence>0.5, frequency, np.nan), linewidth=2)
    plt.yscale('log')
    frequency, confidence = refine(frequency, activation, audio_path)
    plt.subplot(2, 1, 2)
    plt.plot(frequency, linewidth=2)
    plt.yscale('log')
    plt.show()