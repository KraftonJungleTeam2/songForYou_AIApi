import whisper

def transcribe_audio(audio_path):
    # Whisper 모델 로드 ("base" 모델 사용)
    model = whisper.load_model("medium")

    # 오디오 파일 전사
    result = model.transcribe(audio_path, language='ko')

    # 전사된 텍스트 반환
    return result["text"]

if __name__ == "__main__":
    audio_path = "vocals love wins all.wav"
    transcription = transcribe_audio(audio_path)
    print("전사된 텍스트:")
    print(transcription)
