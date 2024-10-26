import os
import webrtcvad
import wave
import numpy as np
from pydub import AudioSegment


def vocal_preprocess(audio_path: str) -> bool:
    """
    음정/가사 추출 전 보컬음원의 전처리 담당
    Args:
        audio_path (_str_): 보컬음원이 있는 path

    Returns:
        bool: 정상 처리시 True, 그 외 False 반환
    """
    return remove_noise(audio_path)

def remove_noise(audio_path, sensitivity=2) -> bool:
    """
    목소리를 감지해 목소리가 아닌 부분은 음소거
    Args:
        audio_path (_type_): _description_
        sensitivity (int, optional): 1, 2, 3으로 감도 입력. 3이 가장 민감하게 처리. Defaults to 2.

    Returns:
        bool: _description_
    """    
    # 음성 파일 로드
    audio = AudioSegment.from_file(os.path.join(audio_path, 'vocals.wav'))

    # Mono로 변환 (VAD는 Mono만 지원)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(48000)  # VAD가 지원하는 샘플 레이트로 설정 (8000, 16000, 32000, 48000 중 하나)

    # WAV 파일로 저장하여 WebRTC VAD 적용
    audio.export(os.path.join(audio_path, "vocals_mono.wav"), format="wav")
    
    # WebRTC VAD 설정
    vad = webrtcvad.Vad()
    vad.set_mode(sensitivity)  # 모드 3: 가장 민감한 설정

    # WAV 파일 읽기
    with wave.open(os.path.join(audio_path, "vocals_mono.wav"), 'rb') as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        frame_length = int(sample_rate * 0.03)  # 30ms 단위

    # 음성 데이터를 NumPy 배열로 변환
    samples = np.frombuffer(frames, dtype=np.int16)

    # 음성 구간 탐지
    segments = []
    for i in range(0, len(samples), frame_length):
        frame = samples[i:i + frame_length]
        if len(frame) < frame_length:
            break

        # VAD를 이용해 음성인지 여부 확인
        is_speech = vad.is_speech(frame.tobytes(), sample_rate)
        segments.append(is_speech)

    # 탐지된 음성 구간에 따라 원본 오디오에서 음소거 처리
    output_audio = AudioSegment.silent(duration=len(audio))

    for i, is_speech in enumerate(segments):
        if is_speech:
            # 말하는 구간을 유지하고, 아닌 구간은 음소거 처리
            start = i * 30  # 30ms 단위
            end = (i + 1) * 30
            output_audio = output_audio.overlay(audio[start:end], position=start)

    # 결과 저장
    output_audio.export(os.path.join(audio_path, "vocals_preprocessed.wav"), format="wav")

    return True

if __name__ == '__main__':
    remove_noise("results/love wins all")