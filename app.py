# app.py
# python==3.12.3
# flask==3.0.3
# openai-whisper==20240930
# demucs==4.0.1

# TODO: 비동기처리

from flask import Flask, request, jsonify, send_from_directory
import os
import shutil
from vocal_seperation import separate_audio
from vocal_preprocess import vocal_preprocess
from vocal_pitch import pitch_extract
from vocal_lyrics import transcribe_audio, vocal_align
import psycopg2
from dotenv import load_dotenv
from pydub import AudioSegment

app = Flask(__name__)

load_dotenv()

db_user = os.getenv("DB_USER")
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")
db_password = os.getenv("DB_PASSWORD")

# 분리된 오디오 파일이 저장될 폴더 경로 설정
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def wav_to_mp3(wav_path):
    mp3_path = os.path.splitext(wav_path)[0]+'.mp3'
    
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3")
    
    return mp3_path

# '.mp3', 'id' in
# save lyrics, inst., pitchdata
@app.route('/seperate', methods=['POST'])
def upload_file():
    # 오디오 파일이 전달되었는지 확인
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    fileid = request.form.get('id')
    is_public = request.form.get('is_public')

    if not fileid:
        return jsonify({'error': 'id must be provided'}), 400

    
    # 파일 저장
    if file and file.filename and fileid:
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # 고유 ID로 결과 디렉토리 생성
        output_dir = os.path.join(RESULT_FOLDER, str(fileid))

        # Demucs를 사용하여 오디오 분리
        if not separate_audio(file_path, output_dir):
            return jsonify({'error': 'error occured while separating audio'}), 400
        if not vocal_preprocess(output_dir):
            return jsonify({'error': 'error occured while preprocessing vocals.wav'}), 400
        if not pitch_extract(output_dir):
            return jsonify({'error': "error occured whlie extracting pitch"}), 400
        if not transcribe_audio(output_dir):
            return jsonify({'error': "error occured whlie trancribing audio"}), 400

        cur = conn.cursor()


        # 원본 파일 읽기
        with open(file_path, 'rb') as f:
            original = f.read()
        # inst. 파일 읽기
        with open(wav_to_mp3(os.path.join(output_dir, "no_vocals.wav")), 'rb') as f:
            no_vocals = f.read()
        # vocal 파일 읽기
        with open(wav_to_mp3(os.path.join(output_dir, "vocals.wav")), 'rb') as f:
            vocals = f.read()

        # SQL 삽입 쿼리
        insert_query = """
            INSERT INTO user_songs (user_id, song, data)
            VALUES (%s, %s, %s)
        """

        # 데이터 삽입 (BLOB 데이터는 psycopg2.Binary로 감싸서 처리)
        cur.execute(insert_query, (1, psycopg2.Binary(song_data), psycopg2.Binary(data_blob)))

        # 변경 사항 커밋
        conn.commit()

        cur.close()
        
        return jsonify({'id': fileid, 'msg': 'process success', 'key': key})
    
    return jsonify({'error': 'File could not be processed'}), 400

if __name__ == '__main__':
    conn = psycopg2.connect(host=db_host, database=db_name, user=db_user, password=db_password)
    app.run(debug=True, host='0.0.0.0', port=5000)
    conn.close()
