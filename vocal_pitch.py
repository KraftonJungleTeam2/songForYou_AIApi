import librosa
# import crepe
from crepe.core import model_srate, as_strided, to_viterbi_cents, to_local_average_cents, build_and_load_model
import numpy as np
import pandas as pd
import os
from scipy.ndimage import gaussian_filter

model = build_and_load_model('full')
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
    audio, sr = librosa.load(os.path.join(audio_path, "vocals_preprocessed.wav"))
    # Crepe로 음정 추출
    _, frequency, _, activation = predict(audio, sr, viterbi=True, step_size=step_size)
    # TODO: lyrics 형식 바뀜
    # if lyrics:
    #     lyrics_masking(activation, lyrics, step_size)
    frequency, confidence = refine(frequency, activation)
    
    np.save(os.path.join(audio_path, "frequency.npy"), frequency)
    np.save(os.path.join(audio_path, "confidence.npy"), confidence)
    np.save(os.path.join(audio_path, "activation.npy"), activation)

    return frequency, confidence, activation

def lyrics_masking(data: np.ndarray, lyrics: dict, step_size: int):
    last = 0
    for s, e in zip(lyrics['start'], lyrics['end']):
        e = int(e*1000//step_size)
        s = int(s*1000//step_size)
        data[:, last:s] = 0
        last = e
    data[:, last:] = 0

def refine(frequency, activation, threshold=0.25):
    # frequency = np.where((frequency >= 40)| (frequency <= 2000), frequency, np.nan)
    new_confidence = distorted_gaussian_filter(activation, 7)
    new_frequency = confidence_filter(smooth(frequency), new_confidence, threshold)
    
    return new_frequency, new_confidence

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

    # lyrics = {"start": [5.45, 24.77, 49.050000000000004, 76.41, 86.83, 97.93], "end": [23.45, 47.13, 76.41, 84.73, 97.92999999999999, 113.53], "text": [" yesterday all my trouble seems so far away now it looks as though they're here to stay oh i believe in yesterday suddenly", " i'm not half the man i used to be there's a shadow hanging over me oh yesterday came suddenly why she had to go i don't know she wouldn't say", " yesterday i said something wrong now i long for yesterday yesterday love was such an easy game to play i need a place to hide away oh i believe in yesterday", " why she had to go i don't know she wouldn't say", " i said something wrong now i long for yesterday yesterday", " yesterday love was such an easy game to play i need a place to hide away oh i believe in yesterday"]}
    # result = pitch_extract(".", step_size=50)
    # if result:
    #     time, frequency, confidence, activation = result
    # time = np.load("results/signal/time.npy")
    # frequency = np.load("results/af5bfed8-ead1-4e48-906f-88132f36e114/frequency.npy")
    # confidence = np.load("results/signal/confidence.npy")
    # activation = np.load("results/signal/activation.npy")
    # print(frequency)
    # plt.figure(figsize=(12, 6))
    audio, sr = librosa.load(os.path.join(".", "vocals_preprocessed.wav"))
    time, frequency, confidence, activation = predict(audio, sr, viterbi=False, step_size=50)
    plt.plot(np.where(confidence > 0.5, frequency, np.nan))
    time, frequency, confidence, activation = predict(audio, sr, viterbi=True, step_size=50)
    plt.plot(np.where(confidence > 0.5, frequency, np.nan))
    plt.yscale('log')
    # plt.pcolor(activation.T)
    plt.show()