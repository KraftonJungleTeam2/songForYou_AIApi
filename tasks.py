import os
import sys
import traceback
import boto3
from celery import Celery
from vocal_separation import separate_audio
from vocal_preprocess import vocal_preprocess
from vocal_pitch import pitch_extract
from vocal_lyrics import transcribe_audio, vocal_align
import psycopg2

# Celery 인스턴스 생성
celery = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# S3 클라이언트 생성
s3 = boto3.client('s3')
bucket_name = "????????????"

RESULT_FOLDER = "results"

@celery.task(bind=True)
def process(self, form, file_name, img_name):
    # 작업 ID로 디렉토리 생성
    task_id = self.request.id
    output_dir = os.path.join(RESULT_FOLDER, task_id)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 처리할 파일 다운로드
        file_path = os.path.join(output_dir, file_name)
        s3.download_file(bucket_name, file_name, file_path)
        img_path = os.path.join(output_dir, img_name)
        s3.download_file(bucket_name, img_name, img_path)
    except:
        return {"msg": "cannot download file", "traceback": traceback.format_exc()}
    
    # Demucs를 사용하여 오디오 분리
    if not separate_audio(file_path, output_dir):
        return {'error': 'error while separating audio'}
    if not vocal_preprocess(output_dir):
        return {'error': 'error while preprocessing vocals.wav'}
    if not (lyrics := transcribe_audio(output_dir)):
        return {'error': "error whlie trancribing audio"}
    if not (pitch_extracted := pitch_extract(output_dir, lyrics)):
        return {'error': "error whlie extracting pitch"}

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
    pitch = pitch_extracted[1].tolist()
    confidence = pitch_extracted[2].tolist()
    with open(os.path.join(output_dir, "activation.npy"), 'rb') as f:
        activation = f.read()
    with open(img_path, 'rb') as f:
        image_file = f.read()


def query():
    # SQL 삽입 쿼리
    insert_query = """
        INSERT INTO songs (user_id, original_song, mr_data, vocal_data, metadata, is_public, pitch, pitch_confidence, pitch_activation, lyrics, genre, image, upload_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        RETURNING id;
    """

    try:
        # 데이터 삽입 (BLOB 데이터는 psycopg2.Binary로 감싸서 처리)
        cur.execute(insert_query, (user_id, 
                                psycopg2.Binary(original), 
                                psycopg2.Binary(no_vocals),
                                psycopg2.Binary(vocals),
                                json.dumps(metadata),
                                is_public,
                                pitch,
                                confidence,
                                psycopg2.Binary(activation),
                                json.dumps(lyrics),
                                genre,
                                psycopg2.Binary(image_file)))

        # 변경 사항 커밋
        if not (row_id := cur.fetchone()):
            return jsonify({'error': 'Error while processing'})
        row_id = row_id[0]
    except Exception as e:
        print(e)
        conn.rollback()
        cur.close()
        return jsonify({'error': 'sql execution error'})
    else:
        conn.commit()
        cur.close()

        shutil.rmtree(output_dir)
        
        return jsonify({'id': row_id, 'msg': 'process success', 'key': key})
