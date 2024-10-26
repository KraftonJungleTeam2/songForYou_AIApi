import whisper
import sys
import re
from difflib import SequenceMatcher
import os
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import json


def align_words(transcribed_words, reference_words):
    # 단어들을 소문자로 변환하여 비교
    transcribed_tokens = [re.sub(r'\W+', '', w['word'].lower()) for w in transcribed_words]
    reference_tokens = [re.sub(r'\W+', '', w.lower()) for w in reference_words]

    matcher = SequenceMatcher(None, transcribed_tokens, reference_tokens)
    aligned_words = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for idx in range(i2 - i1):
                word_info = transcribed_words[i1 + idx]
                aligned_words.append({
                    'word': reference_words[j1 + idx],
                    'start': word_info['start'],
                    'end': word_info['end']
                })
        # 필요에 따라 다른 태그 처리 추가
    return aligned_words

def vocal_align(audio_path, transcript_path):
    # 파일 경로 확인
    if not os.path.isfile(audio_path):
        print(f"오디오 파일을 찾을 수 없습니다: {audio_path}")
        sys.exit(1)
    if not os.path.isfile(transcript_path):
        print(f"텍스트 파일을 찾을 수 없습니다: {transcript_path}")
        sys.exit(1)

    # Whisper 모델 로드
    model = whisper.load_model("base")

    # 오디오 파일 전사 (단어별 타임스탬프 포함)
    result = model.transcribe(audio_path, word_timestamps=True)
    transcribed_words = []
    for segment in result['segments']:
        transcribed_words.extend(segment['words'])

    # 전사된 텍스트 파일 읽기
    with open(transcript_path, 'r', encoding='utf-8') as f:
        reference_text = f.read()
    reference_words = re.findall(r'\w+', reference_text)

    # 단어들 정렬
    aligned = align_words(transcribed_words, reference_words)

    # 결과 출력
    for word_info in aligned:
        print(f"{word_info['start']:.2f} - {word_info['end']:.2f}: {word_info['word']}")

def trim_start_silence(audio_path) -> float:
    """오디오 파일의 앞 묵음 부분을 제거하고 덮어씌움

    Args:
        audio_path (str): vocals_preprocessed.wav가 있는 폴더

    Returns:
        float: 잘라낸 시간을 초단위로 반환함
    """    
    audio = AudioSegment.from_file(os.path.join(audio_path, "vocals_preprocessed.wav"))

    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-40)

    if nonsilent_ranges:
        # 첫 번째 묵음이 아닌 구간의 시작점부터 파일 자르기
        start_trim = nonsilent_ranges[0][0] # milliseconds
        
        # 자른 오디오 파일 저장
        trimmed_audio = audio[start_trim:]
        trimmed_audio.export(os.path.join(audio_path, "vocals_preprocessed_trimed.wav"), format="wav")
    else:
        start_trim = 0

    return start_trim / 1000

def transcribe_audio(audio_path, language='ko') -> dict:
    # 앞 30초 무시하는 문제 -> 묵음 제외
    nonsilent_flat = []
    audio = AudioSegment.from_file(os.path.join(audio_path, "vocals_preprocessed.wav"))
    nonsilent = detect_nonsilent(audio, silence_thresh=-50, seek_step=100)
    nonsilent_flat = [i/1000 for r in nonsilent for i in r] # 1차원으로, ms에서 s로 변경

    model = whisper.load_model("medium")
    print("model loaded: ", model)
    
    # 오디오 파일 전사
    file_path = os.path.join(audio_path, "vocals_preprocessed.wav")
    result = model.transcribe(file_path, language=language, word_timestamps=True, temperature=0, clip_timestamps=nonsilent_flat)

    print("saving results")
    # start, end, text, words = [], [], [], []
    # for seg in result['segments']:
    #     start.append(seg['start'])
    #     end.append(seg['end'])
    #     text.append(seg['text'])
    #     words.append(seg['words'])
    with open('lyrics.json', 'w') as f:
        json.dump(result, f)

    # 전사된 텍스트 반환
    return result


if __name__ == '__main__':
    print(transcribe_audio('results/love wins all'))