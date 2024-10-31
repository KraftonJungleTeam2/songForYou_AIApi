import librosa
# import crepe
from crepe.core import model_srate, as_strided, to_viterbi_cents, to_local_average_cents, build_and_load_model
import numpy as np
import pandas as pd
import os
from scipy.ndimage import gaussian_filter

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
    audio, sr = librosa.load(os.path.join(audio_path, "vocals_preprocessed.wav"))
    # Crepe로 음정 추출
    _, frequency, _, activation = predict(audio, sr, viterbi=True, step_size=step_size)
    frequency, confidence = refine(frequency, activation, lyrics)
    
    np.save(os.path.join(audio_path, "frequency.npy"), frequency)
    np.save(os.path.join(audio_path, "confidence.npy"), confidence)
    np.save(os.path.join(audio_path, "activation.npy"), activation)

    return frequency, confidence, activation

def refine(frequency, activation, lyrics = None, threshold=0.25):
    # frequency 이상치 제거
    Q1 = np.nanpercentile(frequency, 25)
    Q3 = np.nanpercentile(frequency, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    frequency = np.where(frequency <= upper_bound, frequency, np.nan)
    frequency = np.where(frequency >= lower_bound, frequency, np.nan)

    # 가사로 마스킹 => 좋은 생각이 아닌 듯
    # lyrics_masking(activation, lyrics)
    # 가우시안 필터
    new_confidence = distorted_gaussian_filter(activation, 7)
    # 스무딩 & new_frequency 생성
    new_frequency = smooth(confidence_filter(frequency, new_confidence, threshold))
    
    
    return new_frequency, new_confidence

def lyrics_masking(data: np.ndarray, lyrics: dict):
    
    segs = lyrics.get('segments', [])
    last = 0
    for seg in segs:
        temp = int(seg.get('start', 0)*1000/50)
        data[last:temp, :] = 0
        last = int(seg.get('end', 0)*1000/50)
    if last != 0:
        data[last:, :] = 0

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
                print(frequency_[i-down_streak-1], frequency_[i-1]  , (down_streak) )
                frequency[i-down_streak-1:i] = np.nan
            up_streak = 0
            down_streak = 0
        elif v > last:
            if down_streak > 0 and (down_streak > 2 or frequency_[i-down_streak-1]/frequency_[i-1] ** (1/down_streak) > 1.33):
                print(frequency_[i-down_streak-1], frequency_[i-1]  , (down_streak) )
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
    # audio, sr = librosa.load(os.path.join(".", "vocals_preprocessed.wav"))
    # time, frequency, confidence, activation = predict(audio, sr, viterbi=False, step_size=50)
    # plt.plot(np.where(confidence > 0.5, frequency, np.nan))
    # time, frequency, confidence, activation = predict(audio, sr, viterbi=True, step_size=50)
    # plt.plot(np.where(confidence > 0.5, frequency, np.nan))
    # plt.yscale('log')
    # # plt.pcolor(activation.T)
    # plt.show()
    from crepe.core import to_viterbi_cents
    lyrics = {"text": " Slip inside the eye of your mind Don't you know you might find A better place to play You said that you'd never been But all the things that you've seen Will slowly fade away So I start a revolution from my bed Cause you set the brains I had way to my head Step outside, summertime's in bloom Stand up beside the fireplace Take that look from off your face You ain't ever gonna burn my heart out And so Sally can wait She knows it's too late as we're walking on by Her soul slides away But don't look back in anger I heard you say Take me to the place where you go Where nobody knows If it's night or day But please don't put your life in the hands Of a rock and roll band We'll throw it all away I'm gonna start a revolution from my bed Cause you set the brains I had way to my head Step outside, cause summertime's in bloom Stand up beside the fireplace Take that look from off your face Cause you ain't ever gonna burn my heart out And so Sally can wait She knows it's too late as she's walking on by My soul slides away But don't look back in anger I heard you say And so Sally can wait She knows it's too late as we're walking on by Her soul slides away But don't look back in anger I heard you say And so Sally can wait She knows it's too late as she's walking on by My soul slides away But don't look back in anger Don't look back in anger I heard you say At least not today", "segments": [{"id": 0, "seek": 1180, "start": 11.8, "end": 15.78, "text": " Slip inside the eye of your mind", "tokens": [50364, 6187, 647, 1854, 264, 3313, 295, 428, 1575, 50564], "temperature": 0, "avg_logprob": -0.3189259281864873, "compression_ratio": 1.0625, "no_speech_prob": 0.1847728192806244, "words": [{"word": " Slip", "start": 11.8, "end": 12.48, "probability": 0.6395775228738785}, {"word": " inside", "start": 12.48, "end": 13.26, "probability": 0.9700613021850586}, {"word": " the", "start": 13.26, "end": 13.72, "probability": 0.9490077495574951}, {"word": " eye", "start": 13.72, "end": 13.94, "probability": 0.980448305606842}, {"word": " of", "start": 13.94, "end": 14.22, "probability": 0.9934808015823364}, {"word": " your", "start": 14.22, "end": 14.74, "probability": 0.9786537289619446}, {"word": " mind", "start": 14.74, "end": 15.78, "probability": 0.9875839352607727}]}, {"id": 1, "seek": 1180, "start": 15.78, "end": 19.2, "text": " Don't you know you might find", "tokens": [50564, 1468, 380, 291, 458, 291, 1062, 915, 50714], "temperature": 0, "avg_logprob": -0.3189259281864873, "compression_ratio": 1.0625, "no_speech_prob": 0.1847728192806244, "words": [{"word": " Don't", "start": 15.78, "end": 16.36, "probability": 0.8613297045230865}, {"word": " you", "start": 16.36, "end": 16.5, "probability": 0.9687244296073914}, {"word": " know", "start": 16.5, "end": 16.76, "probability": 0.9966639876365662}, {"word": " you", "start": 16.76, "end": 16.98, "probability": 0.9086695313453674}, {"word": " might", "start": 16.98, "end": 17.68, "probability": 0.986748218536377}, {"word": " find", "start": 17.68, "end": 19.2, "probability": 0.9936860203742981}]}, {"id": 2, "seek": 1180, "start": 19.2, "end": 21.36, "text": " A better place to play", "tokens": [50714, 316, 1101, 1081, 281, 862, 50864], "temperature": 0, "avg_logprob": -0.3189259281864873, "compression_ratio": 1.0625, "no_speech_prob": 0.1847728192806244, "words": [{"word": " A", "start": 19.2, "end": 19.9, "probability": 0.6950988173484802}, {"word": " better", "start": 19.9, "end": 20.1, "probability": 0.9909381866455078}, {"word": " place", "start": 20.1, "end": 20.5, "probability": 0.9988069534301758}, {"word": " to", "start": 20.5, "end": 20.9, "probability": 0.998773992061615}, {"word": " play", "start": 20.9, "end": 21.36, "probability": 0.9515995383262634}]}, {"id": 3, "seek": 2360, "start": 23.6, "end": 27.46, "text": " You said that you'd never been", "tokens": [50364, 509, 848, 300, 291, 1116, 1128, 668, 50564], "temperature": 0, "avg_logprob": -0.10925625837766208, "compression_ratio": 1.1, "no_speech_prob": 0.32934001088142395, "words": [{"word": " You", "start": 23.6, "end": 24.22, "probability": 0.43728864192962646}, {"word": " said", "start": 24.22, "end": 24.96, "probability": 0.957190752029419}, {"word": " that", "start": 24.96, "end": 25.34, "probability": 0.9024587869644165}, {"word": " you'd", "start": 25.34, "end": 25.76, "probability": 0.9210655093193054}, {"word": " never", "start": 25.76, "end": 26.38, "probability": 0.9938268065452576}, {"word": " been", "start": 26.38, "end": 27.46, "probability": 0.5291470289230347}]}, {"id": 4, "seek": 2360, "start": 27.46, "end": 30.98, "text": " But all the things that you've seen", "tokens": [50564, 583, 439, 264, 721, 300, 291, 600, 1612, 50714], "temperature": 0, "avg_logprob": -0.10925625837766208, "compression_ratio": 1.1, "no_speech_prob": 0.32934001088142395, "words": [{"word": " But", "start": 27.46, "end": 27.98, "probability": 0.560639500617981}, {"word": " all", "start": 27.98, "end": 28.18, "probability": 0.9893590211868286}, {"word": " the", "start": 28.18, "end": 28.4, "probability": 0.9471266865730286}, {"word": " things", "start": 28.4, "end": 28.62, "probability": 0.9953157901763916}, {"word": " that", "start": 28.62, "end": 29.08, "probability": 0.9715374708175659}, {"word": " you've", "start": 29.08, "end": 29.6, "probability": 0.9747499525547028}, {"word": " seen", "start": 29.6, "end": 30.98, "probability": 0.993196964263916}]}, {"id": 5, "seek": 2360, "start": 30.98, "end": 33.34, "text": " Will slowly fade away", "tokens": [50714, 3099, 5692, 21626, 1314, 50864], "temperature": 0, "avg_logprob": -0.10925625837766208, "compression_ratio": 1.1, "no_speech_prob": 0.32934001088142395, "words": [{"word": " Will", "start": 30.98, "end": 31.68, "probability": 0.4078514277935028}, {"word": " slowly", "start": 31.68, "end": 31.9, "probability": 0.9958714842796326}, {"word": " fade", "start": 31.9, "end": 32.44, "probability": 0.9966707825660706}, {"word": " away", "start": 32.44, "end": 33.34, "probability": 0.9970657229423523}]}, {"id": 6, "seek": 3580, "start": 35.8, "end": 39.6, "text": " So I start a revolution from my bed", "tokens": [50364, 407, 286, 722, 257, 8894, 490, 452, 2901, 50564], "temperature": 0, "avg_logprob": -0.1000017903067849, "compression_ratio": 0.813953488372093, "no_speech_prob": 0.32859930396080017, "words": [{"word": " So", "start": 35.8, "end": 36.22, "probability": 0.5778618454933167}, {"word": " I", "start": 36.22, "end": 36.46, "probability": 0.891933023929596}, {"word": " start", "start": 36.46, "end": 36.64, "probability": 0.8208051919937134}, {"word": " a", "start": 36.64, "end": 36.92, "probability": 0.9915127158164978}, {"word": " revolution", "start": 36.92, "end": 37.56, "probability": 0.9864612221717834}, {"word": " from", "start": 37.56, "end": 38.28, "probability": 0.9754908680915833}, {"word": " my", "start": 38.28, "end": 38.92, "probability": 0.9226180911064148}, {"word": " bed", "start": 38.92, "end": 39.6, "probability": 0.6963574886322021}]}, {"id": 7, "seek": 4110, "start": 41.1, "end": 45.52, "text": " Cause you set the brains I had way to my head", "tokens": [50364, 10865, 291, 992, 264, 15442, 286, 632, 636, 281, 452, 1378, 50564], "temperature": 0, "avg_logprob": -0.31436072077069965, "compression_ratio": 0.8490566037735849, "no_speech_prob": 0.6603372097015381, "words": [{"word": " Cause", "start": 41.1, "end": 41.32, "probability": 0.16054946184158325}, {"word": " you", "start": 41.32, "end": 41.64, "probability": 0.9609706997871399}, {"word": " set", "start": 41.64, "end": 41.9, "probability": 0.7969955801963806}, {"word": " the", "start": 41.9, "end": 42.22, "probability": 0.9959314465522766}, {"word": " brains", "start": 42.22, "end": 42.64, "probability": 0.9597395062446594}, {"word": " I", "start": 42.64, "end": 43.08, "probability": 0.4443807899951935}, {"word": " had", "start": 43.08, "end": 43.42, "probability": 0.9768533706665039}, {"word": " way", "start": 43.42, "end": 43.7, "probability": 0.17580318450927734}, {"word": " to", "start": 43.7, "end": 44.12, "probability": 0.8719877004623413}, {"word": " my", "start": 44.12, "end": 44.9, "probability": 0.9794904589653015}, {"word": " head", "start": 44.9, "end": 45.52, "probability": 0.8054856061935425}]}, {"id": 8, "seek": 4720, "start": 47.2, "end": 51.22, "text": " Step outside, summertime's in bloom", "tokens": [50364, 5470, 2380, 11, 43785, 311, 294, 26899, 50564], "temperature": 0, "avg_logprob": -0.34180521965026855, "compression_ratio": 0.813953488372093, "no_speech_prob": 0.5740166306495667, "words": [{"word": " Step", "start": 47.2, "end": 47.64, "probability": 0.49926093220710754}, {"word": " outside,", "start": 47.64, "end": 48.84, "probability": 0.9900276064872742}, {"word": " summertime's", "start": 49.42, "end": 50.34, "probability": 0.7508408725261688}, {"word": " in", "start": 50.34, "end": 50.54, "probability": 0.9286860823631287}, {"word": " bloom", "start": 50.54, "end": 51.22, "probability": 0.488246887922287}]}, {"id": 9, "seek": 5310, "start": 53.1, "end": 55.72, "text": " Stand up beside the fireplace", "tokens": [50364, 9133, 493, 15726, 264, 39511, 50514], "temperature": 0, "avg_logprob": -0.11039653846195766, "compression_ratio": 1.1590909090909092, "no_speech_prob": 0.6731230020523071, "words": [{"word": " Stand", "start": 53.1, "end": 53.56, "probability": 0.14100874960422516}, {"word": " up", "start": 53.56, "end": 53.92, "probability": 0.9701151251792908}, {"word": " beside", "start": 53.92, "end": 54.4, "probability": 0.8452746868133545}, {"word": " the", "start": 54.4, "end": 54.8, "probability": 0.985689640045166}, {"word": " fireplace", "start": 54.8, "end": 55.72, "probability": 0.9791441559791565}]}, {"id": 10, "seek": 5310, "start": 55.72, "end": 58.72, "text": " Take that look from off your face", "tokens": [50514, 3664, 300, 574, 490, 766, 428, 1851, 50664], "temperature": 0, "avg_logprob": -0.11039653846195766, "compression_ratio": 1.1590909090909092, "no_speech_prob": 0.6731230020523071, "words": [{"word": " Take", "start": 55.72, "end": 56.64, "probability": 0.6099167466163635}, {"word": " that", "start": 56.64, "end": 57.0, "probability": 0.9877845048904419}, {"word": " look", "start": 57.0, "end": 57.28, "probability": 0.941157341003418}, {"word": " from", "start": 57.28, "end": 57.68, "probability": 0.9791203737258911}, {"word": " off", "start": 57.68, "end": 57.94, "probability": 0.9725654125213623}, {"word": " your", "start": 57.94, "end": 58.22, "probability": 0.9514098167419434}, {"word": " face", "start": 58.22, "end": 58.72, "probability": 0.994916558265686}]}, {"id": 11, "seek": 5310, "start": 58.72, "end": 65.04, "text": " You ain't ever gonna burn my heart out", "tokens": [50664, 509, 7862, 380, 1562, 799, 5064, 452, 1917, 484, 51014], "temperature": 0, "avg_logprob": -0.11039653846195766, "compression_ratio": 1.1590909090909092, "no_speech_prob": 0.6731230020523071, "words": [{"word": " You", "start": 58.72, "end": 59.64, "probability": 0.9535019397735596}, {"word": " ain't", "start": 59.64, "end": 59.96, "probability": 0.9797947108745575}, {"word": " ever", "start": 59.96, "end": 60.58, "probability": 0.8947926759719849}, {"word": " gonna", "start": 60.58, "end": 61.26, "probability": 0.9479248523712158}, {"word": " burn", "start": 61.26, "end": 61.68, "probability": 0.5980392098426819}, {"word": " my", "start": 61.68, "end": 62.48, "probability": 0.9747783541679382}, {"word": " heart", "start": 62.48, "end": 62.9, "probability": 0.9915286302566528}, {"word": " out", "start": 62.9, "end": 65.04, "probability": 0.9632464647293091}]}, {"id": 12, "seek": 7060, "start": 70.6, "end": 74.52, "text": " And so Sally can wait", "tokens": [50364, 400, 370, 26385, 393, 1699, 50564], "temperature": 0, "avg_logprob": -0.0917544039812955, "compression_ratio": 0.9855072463768116, "no_speech_prob": 0.48193249106407166, "words": [{"word": " And", "start": 70.6, "end": 71.22, "probability": 0.5509408712387085}, {"word": " so", "start": 71.22, "end": 72.44, "probability": 0.9820797443389893}, {"word": " Sally", "start": 72.44, "end": 73.08, "probability": 0.6879513263702393}, {"word": " can", "start": 73.08, "end": 73.66, "probability": 0.9240266680717468}, {"word": " wait", "start": 73.66, "end": 74.52, "probability": 0.904793381690979}]}, {"id": 13, "seek": 7060, "start": 74.52, "end": 80.54, "text": " She knows it's too late as we're walking on by", "tokens": [50564, 1240, 3255, 309, 311, 886, 3469, 382, 321, 434, 4494, 322, 538, 50864], "temperature": 0, "avg_logprob": -0.0917544039812955, "compression_ratio": 0.9855072463768116, "no_speech_prob": 0.48193249106407166, "words": [{"word": " She", "start": 74.52, "end": 75.3, "probability": 0.6176494359970093}, {"word": " knows", "start": 75.3, "end": 75.8, "probability": 0.9848389625549316}, {"word": " it's", "start": 75.8, "end": 76.4, "probability": 0.9752229154109955}, {"word": " too", "start": 76.4, "end": 76.6, "probability": 0.9913643598556519}, {"word": " late", "start": 76.6, "end": 77.3, "probability": 0.9994598031044006}, {"word": " as", "start": 77.3, "end": 77.64, "probability": 0.40164703130722046}, {"word": " we're", "start": 77.64, "end": 78.68, "probability": 0.9841435849666595}, {"word": " walking", "start": 78.68, "end": 79.0, "probability": 0.8973681926727295}, {"word": " on", "start": 79.0, "end": 79.62, "probability": 0.98763507604599}, {"word": " by", "start": 79.62, "end": 80.54, "probability": 0.9975290894508362}]}, {"id": 14, "seek": 8230, "start": 82.3, "end": 86.76, "text": " Her soul slides away", "tokens": [50364, 3204, 5133, 9788, 1314, 50564], "temperature": 0, "avg_logprob": -0.13823363997719504, "compression_ratio": 0.9420289855072463, "no_speech_prob": 0.42903801798820496, "words": [{"word": " Her", "start": 82.3, "end": 83.06, "probability": 0.1647704839706421}, {"word": " soul", "start": 83.06, "end": 84.16, "probability": 0.8302649855613708}, {"word": " slides", "start": 84.16, "end": 85.02, "probability": 0.936160683631897}, {"word": " away", "start": 85.02, "end": 86.76, "probability": 0.995844304561615}]}, {"id": 15, "seek": 8230, "start": 86.76, "end": 90.12, "text": " But don't look back in anger", "tokens": [50564, 583, 500, 380, 574, 646, 294, 10240, 50764], "temperature": 0, "avg_logprob": -0.13823363997719504, "compression_ratio": 0.9420289855072463, "no_speech_prob": 0.42903801798820496, "words": [{"word": " But", "start": 86.76, "end": 88.02, "probability": 0.5533970594406128}, {"word": " don't", "start": 88.02, "end": 88.28, "probability": 0.9852817058563232}, {"word": " look", "start": 88.28, "end": 88.56, "probability": 0.9991472959518433}, {"word": " back", "start": 88.56, "end": 89.06, "probability": 0.9992145299911499}, {"word": " in", "start": 89.06, "end": 89.48, "probability": 0.99434494972229}, {"word": " anger", "start": 89.48, "end": 90.12, "probability": 0.9955920577049255}]}, {"id": 16, "seek": 8230, "start": 90.12, "end": 92.4, "text": " I heard you say", "tokens": [50764, 286, 2198, 291, 584, 50864], "temperature": 0, "avg_logprob": -0.13823363997719504, "compression_ratio": 0.9420289855072463, "no_speech_prob": 0.42903801798820496, "words": [{"word": " I", "start": 90.12, "end": 90.9, "probability": 0.7584968209266663}, {"word": " heard", "start": 90.9, "end": 91.2, "probability": 0.9869629740715027}, {"word": " you", "start": 91.2, "end": 91.52, "probability": 0.9385643005371094}, {"word": " say", "start": 91.52, "end": 92.4, "probability": 0.9924271702766418}]}, {"id": 17, "seek": 10330, "start": 103.3, "end": 107.08, "text": " Take me to the place where you go", "tokens": [50364, 3664, 385, 281, 264, 1081, 689, 291, 352, 50564], "temperature": 0, "avg_logprob": -0.09113959471384685, "compression_ratio": 1.0, "no_speech_prob": 0.5421507954597473, "words": [{"word": " Take", "start": 103.3, "end": 103.74, "probability": 0.4625539183616638}, {"word": " me", "start": 103.74, "end": 104.1, "probability": 0.9959779977798462}, {"word": " to", "start": 104.1, "end": 104.7, "probability": 0.9899455904960632}, {"word": " the", "start": 104.7, "end": 104.94, "probability": 0.9507610201835632}, {"word": " place", "start": 104.94, "end": 105.22, "probability": 0.9972988963127136}, {"word": " where", "start": 105.22, "end": 105.72, "probability": 0.9771111607551575}, {"word": " you", "start": 105.72, "end": 106.06, "probability": 0.9954273700714111}, {"word": " go", "start": 106.06, "end": 107.08, "probability": 0.9923754334449768}]}, {"id": 18, "seek": 10330, "start": 107.08, "end": 110.5, "text": " Where nobody knows", "tokens": [50564, 2305, 5079, 3255, 50714], "temperature": 0, "avg_logprob": -0.09113959471384685, "compression_ratio": 1.0, "no_speech_prob": 0.5421507954597473, "words": [{"word": " Where", "start": 107.08, "end": 107.8, "probability": 0.44035765528678894}, {"word": " nobody", "start": 107.8, "end": 108.76, "probability": 0.9566842317581177}, {"word": " knows", "start": 108.76, "end": 110.5, "probability": 0.9757438898086548}]}, {"id": 19, "seek": 10330, "start": 110.5, "end": 112.98, "text": " If it's night or day", "tokens": [50714, 759, 309, 311, 1818, 420, 786, 50864], "temperature": 0, "avg_logprob": -0.09113959471384685, "compression_ratio": 1.0, "no_speech_prob": 0.5421507954597473, "words": [{"word": " If", "start": 110.5, "end": 111.36, "probability": 0.8551788926124573}, {"word": " it's", "start": 111.36, "end": 111.7, "probability": 0.9711946845054626}, {"word": " night", "start": 111.7, "end": 111.84, "probability": 0.9083483815193176}, {"word": " or", "start": 111.84, "end": 112.22, "probability": 0.9961099028587341}, {"word": " day", "start": 112.22, "end": 112.98, "probability": 0.9926732182502747}]}, {"id": 20, "seek": 11490, "start": 114.9, "end": 118.88, "text": " But please don't put your life in the hands", "tokens": [50364, 583, 1767, 500, 380, 829, 428, 993, 294, 264, 2377, 50564], "temperature": 0, "avg_logprob": -0.11221297856034904, "compression_ratio": 1.1097560975609757, "no_speech_prob": 0.7481410503387451, "words": [{"word": " But", "start": 114.9, "end": 115.32, "probability": 0.4018443524837494}, {"word": " please", "start": 115.32, "end": 115.6, "probability": 0.942348062992096}, {"word": " don't", "start": 115.6, "end": 116.02, "probability": 0.9805091917514801}, {"word": " put", "start": 116.02, "end": 116.36, "probability": 0.9921091198921204}, {"word": " your", "start": 116.36, "end": 116.78, "probability": 0.9771989583969116}, {"word": " life", "start": 116.78, "end": 117.12, "probability": 0.986707329750061}, {"word": " in", "start": 117.12, "end": 117.64, "probability": 0.978967010974884}, {"word": " the", "start": 117.64, "end": 118.22, "probability": 0.9914942979812622}, {"word": " hands", "start": 118.22, "end": 118.88, "probability": 0.9603475332260132}]}, {"id": 21, "seek": 11490, "start": 118.88, "end": 122.18, "text": " Of a rock and roll band", "tokens": [50564, 2720, 257, 3727, 293, 3373, 4116, 50714], "temperature": 0, "avg_logprob": -0.11221297856034904, "compression_ratio": 1.1097560975609757, "no_speech_prob": 0.7481410503387451, "words": [{"word": " Of", "start": 118.88, "end": 119.56, "probability": 0.3731223940849304}, {"word": " a", "start": 119.56, "end": 119.78, "probability": 0.9705613255500793}, {"word": " rock", "start": 119.78, "end": 120.12, "probability": 0.9555213451385498}, {"word": " and", "start": 120.12, "end": 120.4, "probability": 0.42819035053253174}, {"word": " roll", "start": 120.4, "end": 120.72, "probability": 0.9960470795631409}, {"word": " band", "start": 120.72, "end": 122.18, "probability": 0.9857674837112427}]}, {"id": 22, "seek": 11490, "start": 122.18, "end": 124.46, "text": " We'll throw it all away", "tokens": [50714, 492, 603, 3507, 309, 439, 1314, 50864], "temperature": 0, "avg_logprob": -0.11221297856034904, "compression_ratio": 1.1097560975609757, "no_speech_prob": 0.7481410503387451, "words": [{"word": " We'll", "start": 122.18, "end": 123.06, "probability": 0.8014014363288879}, {"word": " throw", "start": 123.06, "end": 123.16, "probability": 0.9923398494720459}, {"word": " it", "start": 123.16, "end": 123.5, "probability": 0.9986554384231567}, {"word": " all", "start": 123.5, "end": 123.8, "probability": 0.995630145072937}, {"word": " away", "start": 123.8, "end": 124.46, "probability": 0.9979143738746643}]}, {"id": 23, "seek": 12710, "start": 127.1, "end": 130.9, "text": " I'm gonna start a revolution from my bed", "tokens": [50364, 286, 478, 799, 722, 257, 8894, 490, 452, 2901, 50564], "temperature": 0, "avg_logprob": -0.04321125646432241, "compression_ratio": 0.8333333333333334, "no_speech_prob": 0.7841430902481079, "words": [{"word": " I'm", "start": 127.1, "end": 127.44, "probability": 0.8233376443386078}, {"word": " gonna", "start": 127.44, "end": 127.64, "probability": 0.9516918659210205}, {"word": " start", "start": 127.64, "end": 128.02, "probability": 0.9942156672477722}, {"word": " a", "start": 128.02, "end": 128.28, "probability": 0.9914209246635437}, {"word": " revolution", "start": 128.28, "end": 128.9, "probability": 0.982573926448822}, {"word": " from", "start": 128.9, "end": 129.5, "probability": 0.9747686982154846}, {"word": " my", "start": 129.5, "end": 130.28, "probability": 0.9638510942459106}, {"word": " bed", "start": 130.28, "end": 130.9, "probability": 0.45567557215690613}]}, {"id": 24, "seek": 13250, "start": 132.5, "end": 136.82, "text": " Cause you set the brains I had way to my head", "tokens": [50364, 10865, 291, 992, 264, 15442, 286, 632, 636, 281, 452, 1378, 50564], "temperature": 0, "avg_logprob": -0.051171490124293735, "compression_ratio": 0.8490566037735849, "no_speech_prob": 0.6796178221702576, "words": [{"word": " Cause", "start": 132.5, "end": 132.76, "probability": 0.14979657530784607}, {"word": " you", "start": 132.76, "end": 133.02, "probability": 0.9277206659317017}, {"word": " set", "start": 133.02, "end": 133.26, "probability": 0.4418303966522217}, {"word": " the", "start": 133.26, "end": 133.6, "probability": 0.9958450198173523}, {"word": " brains", "start": 133.6, "end": 134.0, "probability": 0.9380990862846375}, {"word": " I", "start": 134.0, "end": 134.32, "probability": 0.812877357006073}, {"word": " had", "start": 134.32, "end": 134.68, "probability": 0.9936202168464661}, {"word": " way", "start": 134.68, "end": 134.98, "probability": 0.00035127607407048345}, {"word": " to", "start": 134.98, "end": 135.3, "probability": 0.4981172978878021}, {"word": " my", "start": 135.3, "end": 136.12, "probability": 0.968622624874115}, {"word": " head", "start": 136.12, "end": 136.82, "probability": 0.8004426956176758}]}, {"id": 25, "seek": 13850, "start": 138.5, "end": 142.5, "text": " Step outside, cause summertime's in bloom", "tokens": [50364, 5470, 2380, 11, 3082, 43785, 311, 294, 26899, 50564], "temperature": 0, "avg_logprob": -0.18802926757118918, "compression_ratio": 0.8367346938775511, "no_speech_prob": 0.654595136642456, "words": [{"word": " Step", "start": 138.5, "end": 138.98, "probability": 0.41555550694465637}, {"word": " outside,", "start": 138.98, "end": 139.92, "probability": 0.9893929958343506}, {"word": " cause", "start": 140.22, "end": 140.34, "probability": 0.2787705659866333}, {"word": " summertime's", "start": 140.34, "end": 141.6, "probability": 0.6961264312267303}, {"word": " in", "start": 141.6, "end": 141.84, "probability": 0.9846398830413818}, {"word": " bloom", "start": 141.84, "end": 142.5, "probability": 0.8706810474395752}]}, {"id": 26, "seek": 14440, "start": 144.4, "end": 146.96, "text": " Stand up beside the fireplace", "tokens": [50364, 9133, 493, 15726, 264, 39511, 50514], "temperature": 0, "avg_logprob": -0.04453125904346335, "compression_ratio": 1.1612903225806452, "no_speech_prob": 0.3018317222595215, "words": [{"word": " Stand", "start": 144.4, "end": 144.96, "probability": 0.2533267140388489}, {"word": " up", "start": 144.96, "end": 145.34, "probability": 0.8593118786811829}, {"word": " beside", "start": 145.34, "end": 145.78, "probability": 0.8674024939537048}, {"word": " the", "start": 145.78, "end": 146.18, "probability": 0.9856407642364502}, {"word": " fireplace", "start": 146.18, "end": 146.96, "probability": 0.9679173827171326}]}, {"id": 27, "seek": 14440, "start": 146.96, "end": 150.1, "text": " Take that look from off your face", "tokens": [50514, 3664, 300, 574, 490, 766, 428, 1851, 50664], "temperature": 0, "avg_logprob": -0.04453125904346335, "compression_ratio": 1.1612903225806452, "no_speech_prob": 0.3018317222595215, "words": [{"word": " Take", "start": 146.96, "end": 148.04, "probability": 0.42653176188468933}, {"word": " that", "start": 148.04, "end": 148.38, "probability": 0.9504485726356506}, {"word": " look", "start": 148.38, "end": 148.64, "probability": 0.9900256395339966}, {"word": " from", "start": 148.64, "end": 149.08, "probability": 0.9812983274459839}, {"word": " off", "start": 149.08, "end": 149.32, "probability": 0.9605408906936646}, {"word": " your", "start": 149.32, "end": 149.56, "probability": 0.9511985778808594}, {"word": " face", "start": 149.56, "end": 150.1, "probability": 0.9834250807762146}]}, {"id": 28, "seek": 14440, "start": 150.1, "end": 155.96, "text": " Cause you ain't ever gonna burn my heart out", "tokens": [50664, 10865, 291, 7862, 380, 1562, 799, 5064, 452, 1917, 484, 51014], "temperature": 0, "avg_logprob": -0.04453125904346335, "compression_ratio": 1.1612903225806452, "no_speech_prob": 0.3018317222595215, "words": [{"word": " Cause", "start": 150.1, "end": 150.62, "probability": 0.20778444409370422}, {"word": " you", "start": 150.62, "end": 151.02, "probability": 0.9869838953018188}, {"word": " ain't", "start": 151.02, "end": 151.48, "probability": 0.9740056395530701}, {"word": " ever", "start": 151.48, "end": 151.94, "probability": 0.9320938587188721}, {"word": " gonna", "start": 151.94, "end": 152.66, "probability": 0.9477386474609375}, {"word": " burn", "start": 152.66, "end": 153.18, "probability": 0.8847889304161072}, {"word": " my", "start": 153.18, "end": 153.86, "probability": 0.9567161798477173}, {"word": " heart", "start": 153.86, "end": 154.24, "probability": 0.9728259444236755}, {"word": " out", "start": 154.24, "end": 155.96, "probability": 0.9367235898971558}]}, {"id": 29, "seek": 16210, "start": 162.1, "end": 166.0, "text": " And so Sally can wait", "tokens": [50364, 400, 370, 26385, 393, 1699, 50564], "temperature": 0, "avg_logprob": -0.044242650270462036, "compression_ratio": 0.9855072463768116, "no_speech_prob": 0.47474607825279236, "words": [{"word": " And", "start": 162.1, "end": 162.88, "probability": 0.020336037501692772}, {"word": " so", "start": 162.88, "end": 163.8, "probability": 0.9753310680389404}, {"word": " Sally", "start": 163.8, "end": 164.46, "probability": 0.540869414806366}, {"word": " can", "start": 164.46, "end": 164.98, "probability": 0.5834792852401733}, {"word": " wait", "start": 164.98, "end": 166.0, "probability": 0.9021598696708679}]}, {"id": 30, "seek": 16210, "start": 166.0, "end": 172.06, "text": " She knows it's too late as she's walking on by", "tokens": [50564, 1240, 3255, 309, 311, 886, 3469, 382, 750, 311, 4494, 322, 538, 50864], "temperature": 0, "avg_logprob": -0.044242650270462036, "compression_ratio": 0.9855072463768116, "no_speech_prob": 0.47474607825279236, "words": [{"word": " She", "start": 166.0, "end": 166.72, "probability": 0.5973937511444092}, {"word": " knows", "start": 166.72, "end": 167.2, "probability": 0.9853078126907349}, {"word": " it's", "start": 167.2, "end": 167.78, "probability": 0.9797919690608978}, {"word": " too", "start": 167.78, "end": 168.08, "probability": 0.9922916293144226}, {"word": " late", "start": 168.08, "end": 168.7, "probability": 0.9991543292999268}, {"word": " as", "start": 168.7, "end": 169.06, "probability": 0.12941782176494598}, {"word": " she's", "start": 169.06, "end": 169.66, "probability": 0.9834910035133362}, {"word": " walking", "start": 169.66, "end": 170.32, "probability": 0.8449464440345764}, {"word": " on", "start": 170.32, "end": 171.02, "probability": 0.6698806285858154}, {"word": " by", "start": 171.02, "end": 172.06, "probability": 0.9967619180679321}]}, {"id": 31, "seek": 17360, "start": 173.6, "end": 178.24, "text": " My soul slides away", "tokens": [50364, 1222, 5133, 9788, 1314, 50614], "temperature": 0, "avg_logprob": -0.06513200022957542, "compression_ratio": 0.9411764705882353, "no_speech_prob": 0.4421099126338959, "words": [{"word": " My", "start": 173.6, "end": 174.14, "probability": 0.4759417176246643}, {"word": " soul", "start": 174.14, "end": 175.5, "probability": 0.9591147899627686}, {"word": " slides", "start": 175.5, "end": 176.38, "probability": 0.8915759325027466}, {"word": " away", "start": 176.38, "end": 178.24, "probability": 0.9955641031265259}]}, {"id": 32, "seek": 17360, "start": 178.24, "end": 181.54, "text": " But don't look back in anger", "tokens": [50614, 583, 500, 380, 574, 646, 294, 10240, 50764], "temperature": 0, "avg_logprob": -0.06513200022957542, "compression_ratio": 0.9411764705882353, "no_speech_prob": 0.4421099126338959, "words": [{"word": " But", "start": 178.24, "end": 179.4, "probability": 0.511971116065979}, {"word": " don't", "start": 179.4, "end": 179.82, "probability": 0.9897951185703278}, {"word": " look", "start": 179.82, "end": 179.94, "probability": 0.9945791959762573}, {"word": " back", "start": 179.94, "end": 180.46, "probability": 0.9996023774147034}, {"word": " in", "start": 180.46, "end": 180.86, "probability": 0.9918783903121948}, {"word": " anger", "start": 180.86, "end": 181.54, "probability": 0.9947059750556946}]}, {"id": 33, "seek": 17360, "start": 181.54, "end": 183.8, "text": " I heard you say", "tokens": [50764, 286, 2198, 291, 584, 50864], "temperature": 0, "avg_logprob": -0.06513200022957542, "compression_ratio": 0.9411764705882353, "no_speech_prob": 0.4421099126338959, "words": [{"word": " I", "start": 181.54, "end": 182.34, "probability": 0.8135983943939209}, {"word": " heard", "start": 182.34, "end": 182.62, "probability": 0.9874505996704102}, {"word": " you", "start": 182.62, "end": 182.98, "probability": 0.9528194069862366}, {"word": " say", "start": 182.98, "end": 183.8, "probability": 0.9873884320259094}]}, {"id": 34, "seek": 22110, "start": 221.42000000000002, "end": 224.92, "text": " And so Sally can wait", "tokens": [50364, 400, 370, 26385, 393, 1699, 50564], "temperature": 0, "avg_logprob": -0.0888825678357891, "compression_ratio": 1.368421052631579, "no_speech_prob": 0.3283308744430542, "words": [{"word": " And", "start": 221.42000000000002, "end": 222.74, "probability": 0.0009695412591099739}, {"word": " so", "start": 222.74, "end": 222.74, "probability": 0.1512279212474823}, {"word": " Sally", "start": 222.74, "end": 223.38, "probability": 0.7601997256278992}, {"word": " can", "start": 223.38, "end": 224.04, "probability": 0.9080559611320496}, {"word": " wait", "start": 224.04, "end": 224.92, "probability": 0.9162262082099915}]}, {"id": 35, "seek": 22110, "start": 224.92, "end": 231.22, "text": " She knows it's too late as we're walking on by", "tokens": [50564, 1240, 3255, 309, 311, 886, 3469, 382, 321, 434, 4494, 322, 538, 50914], "temperature": 0, "avg_logprob": -0.0888825678357891, "compression_ratio": 1.368421052631579, "no_speech_prob": 0.3283308744430542, "words": [{"word": " She", "start": 224.92, "end": 225.68, "probability": 0.45453163981437683}, {"word": " knows", "start": 225.68, "end": 226.16, "probability": 0.9810760021209717}, {"word": " it's", "start": 226.16, "end": 226.52, "probability": 0.9624402225017548}, {"word": " too", "start": 226.52, "end": 227.04, "probability": 0.992183268070221}, {"word": " late", "start": 227.04, "end": 227.6, "probability": 0.9988441467285156}, {"word": " as", "start": 227.6, "end": 227.98, "probability": 0.29505452513694763}, {"word": " we're", "start": 227.98, "end": 228.76, "probability": 0.9813068509101868}, {"word": " walking", "start": 228.76, "end": 229.26, "probability": 0.9337869882583618}, {"word": " on", "start": 229.26, "end": 229.98, "probability": 0.9825131893157959}, {"word": " by", "start": 229.98, "end": 231.22, "probability": 0.9970065951347351}]}, {"id": 36, "seek": 22110, "start": 232.1, "end": 237.34, "text": " Her soul slides away", "tokens": [50914, 3204, 5133, 9788, 1314, 51164], "temperature": 0, "avg_logprob": -0.0888825678357891, "compression_ratio": 1.368421052631579, "no_speech_prob": 0.3283308744430542, "words": [{"word": " Her", "start": 232.1, "end": 233.22, "probability": 0.8802509903907776}, {"word": " soul", "start": 233.22, "end": 234.5, "probability": 0.9393745064735413}, {"word": " slides", "start": 234.5, "end": 235.34, "probability": 0.9693686366081238}, {"word": " away", "start": 235.34, "end": 237.34, "probability": 0.9955109357833862}]}, {"id": 37, "seek": 22110, "start": 237.34, "end": 240.4, "text": " But don't look back in anger", "tokens": [51164, 583, 500, 380, 574, 646, 294, 10240, 51364], "temperature": 0, "avg_logprob": -0.0888825678357891, "compression_ratio": 1.368421052631579, "no_speech_prob": 0.3283308744430542, "words": [{"word": " But", "start": 237.34, "end": 238.32, "probability": 0.8742575645446777}, {"word": " don't", "start": 238.32, "end": 238.66, "probability": 0.992646336555481}, {"word": " look", "start": 238.66, "end": 238.92, "probability": 0.9950987696647644}, {"word": " back", "start": 238.92, "end": 239.42, "probability": 0.9992750287055969}, {"word": " in", "start": 239.42, "end": 239.78, "probability": 0.9801511764526367}, {"word": " anger", "start": 239.78, "end": 240.4, "probability": 0.9983235001564026}]}, {"id": 38, "seek": 22110, "start": 241.1, "end": 243.18, "text": " I heard you say", "tokens": [51364, 286, 2198, 291, 584, 51514], "temperature": 0, "avg_logprob": -0.0888825678357891, "compression_ratio": 1.368421052631579, "no_speech_prob": 0.3283308744430542, "words": [{"word": " I", "start": 240.56, "end": 241.22, "probability": 0.7411590814590454}, {"word": " heard", "start": 241.22, "end": 241.54, "probability": 0.9876761436462402}, {"word": " you", "start": 241.54, "end": 241.92, "probability": 0.9438568353652954}, {"word": " say", "start": 241.92, "end": 243.18, "probability": 0.9792554974555969}]}, {"id": 39, "seek": 22110, "start": 244.1, "end": 248.5, "text": " And so Sally can wait", "tokens": [51514, 400, 370, 26385, 393, 1699, 51714], "temperature": 0, "avg_logprob": -0.0888825678357891, "compression_ratio": 1.368421052631579, "no_speech_prob": 0.3283308744430542, "words": [{"word": " And", "start": 244.1, "end": 245.72, "probability": 0.306631475687027}, {"word": " so", "start": 245.72, "end": 246.3, "probability": 0.988802969455719}, {"word": " Sally", "start": 246.3, "end": 247.0, "probability": 0.9925715327262878}, {"word": " can", "start": 247.0, "end": 247.64, "probability": 0.9969627261161804}, {"word": " wait", "start": 247.64, "end": 248.5, "probability": 0.9982935786247253}]}, {"id": 40, "seek": 24850, "start": 248.5, "end": 254.5, "text": " She knows it's too late as she's walking on by", "tokens": [50364, 1240, 3255, 309, 311, 886, 3469, 382, 750, 311, 4494, 322, 538, 50664], "temperature": 0, "avg_logprob": -0.03641451597213745, "compression_ratio": 0.8679245283018868, "no_speech_prob": 0.03159337863326073, "words": [{"word": " She", "start": 248.5, "end": 249.24, "probability": 0.6352056860923767}, {"word": " knows", "start": 249.24, "end": 249.72, "probability": 0.9800455570220947}, {"word": " it's", "start": 249.72, "end": 250.34, "probability": 0.9654598832130432}, {"word": " too", "start": 250.34, "end": 250.6, "probability": 0.987949013710022}, {"word": " late", "start": 250.6, "end": 251.26, "probability": 0.9876437187194824}, {"word": " as", "start": 251.26, "end": 251.6, "probability": 0.0928208976984024}, {"word": " she's", "start": 251.6, "end": 252.36, "probability": 0.9670564830303192}, {"word": " walking", "start": 252.36, "end": 252.86, "probability": 0.921006977558136}, {"word": " on", "start": 252.86, "end": 253.58, "probability": 0.6837833523750305}, {"word": " by", "start": 253.58, "end": 254.5, "probability": 0.9819170832633972}]}, {"id": 41, "seek": 25620, "start": 256.2, "end": 260.96, "text": " My soul slides away", "tokens": [50364, 1222, 5133, 9788, 1314, 50614], "temperature": 0, "avg_logprob": -0.04008376101652781, "compression_ratio": 1.2166666666666666, "no_speech_prob": 0.5507122278213501, "words": [{"word": " My", "start": 256.2, "end": 256.84, "probability": 0.3380844295024872}, {"word": " soul", "start": 256.84, "end": 258.08, "probability": 0.9598000049591064}, {"word": " slides", "start": 258.08, "end": 258.94, "probability": 0.8821771144866943}, {"word": " away", "start": 258.94, "end": 260.96, "probability": 0.9926262497901917}]}, {"id": 42, "seek": 25620, "start": 260.96, "end": 264.14, "text": " But don't look back in anger", "tokens": [50614, 583, 500, 380, 574, 646, 294, 10240, 50764], "temperature": 0, "avg_logprob": -0.04008376101652781, "compression_ratio": 1.2166666666666666, "no_speech_prob": 0.5507122278213501, "words": [{"word": " But", "start": 260.96, "end": 261.98, "probability": 0.4361259937286377}, {"word": " don't", "start": 261.98, "end": 262.26, "probability": 0.9789531528949738}, {"word": " look", "start": 262.26, "end": 262.46, "probability": 0.9976720213890076}, {"word": " back", "start": 262.46, "end": 263.02, "probability": 0.9995583891868591}, {"word": " in", "start": 263.02, "end": 263.46, "probability": 0.9926775693893433}, {"word": " anger", "start": 263.46, "end": 264.14, "probability": 0.9935988187789917}]}, {"id": 43, "seek": 25620, "start": 264.14, "end": 267.24, "text": " Don't look back in anger", "tokens": [50764, 1468, 380, 574, 646, 294, 10240, 50914], "temperature": 0, "avg_logprob": -0.04008376101652781, "compression_ratio": 1.2166666666666666, "no_speech_prob": 0.5507122278213501, "words": [{"word": " Don't", "start": 264.14, "end": 265.26, "probability": 0.9047404527664185}, {"word": " look", "start": 265.26, "end": 265.5, "probability": 0.9990407824516296}, {"word": " back", "start": 265.5, "end": 266.02, "probability": 0.9993000030517578}, {"word": " in", "start": 266.02, "end": 266.42, "probability": 0.9989787340164185}, {"word": " anger", "start": 266.42, "end": 267.24, "probability": 0.9989915490150452}]}, {"id": 44, "seek": 27050, "start": 270.5, "end": 271.86, "text": " I heard you say", "tokens": [50364, 286, 2198, 291, 584, 50464], "temperature": 0, "avg_logprob": -0.09021017381123134, "compression_ratio": 0.6521739130434783, "no_speech_prob": 0.29990339279174805, "words": [{"word": " I", "start": 270.5, "end": 270.82, "probability": 0.5631121397018433}, {"word": " heard", "start": 270.82, "end": 271.08, "probability": 0.9325012564659119}, {"word": " you", "start": 271.08, "end": 271.38, "probability": 0.9758586883544922}, {"word": " say", "start": 271.38, "end": 271.86, "probability": 0.9621339440345764}]}, {"id": 45, "seek": 27920, "start": 279.2, "end": 280.98, "text": " At least not today", "tokens": [50364, 1711, 1935, 406, 965, 50464], "temperature": 0, "avg_logprob": -0.14232595477785384, "compression_ratio": 0.6923076923076923, "no_speech_prob": 0.6412345767021179, "words": [{"word": " At", "start": 279.2, "end": 279.64, "probability": 0.21308422088623047}, {"word": " least", "start": 279.64, "end": 279.92, "probability": 0.9569147825241089}, {"word": " not", "start": 279.92, "end": 280.42, "probability": 0.9812147617340088}, {"word": " today", "start": 280.42, "end": 280.98, "probability": 0.9833679795265198}]}], "language": "en"}
    activation = np.load("test/activation (1).npy")
    frequency = 10 * 2 ** (to_viterbi_cents(activation) / 1200)
    plt.plot(frequency)
    frequency, confidence = refine(frequency, activation, lyrics)
    print(frequency[1450:1475])
    plt.plot(frequency)
    plt.yscale('log')
    plt.show()