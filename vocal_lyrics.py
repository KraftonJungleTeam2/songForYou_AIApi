import whisper
import sys
import re
from difflib import SequenceMatcher
import os

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

def transcribe_audio(audio_path):
    # TODO: 앞 30초 무시하는 문제 있음. 트림으로 해결
    # Whisper 모델 로드 ("base" 모델 사용)
    model = whisper.load_model("medium")

    # 오디오 파일 전사
    result = model.transcribe(audio_path, language='ko', word_timestamps=True, temperature=0, no_speech_threshold=0.3)

    # text = [seg['text'] for seg in result['segments']]
    # time = [(seg['start'], seg['end']) for seg in result['segments']]

    with open(os.path.join(audio_path, 'lyrics.txt'), 'w') as f:
        for seg in result['segments']:
            start, end, text = seg['start'], seg['end'], seg['text']
            f.write(f"{start} {end} {text}\n")

    # 전사된 텍스트 반환
    return True
