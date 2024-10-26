import os
import traceback
import boto3
from celery import Celery, Task
from vocal_separation import separate_audio
from vocal_preprocess import vocal_preprocess
from vocal_lyrics import transcribe_audio
from vocal_pitch import pitch_extract
import psycopg2
from dotenv import load_dotenv
import json
import numpy as np
from pydub import AudioSegment
import time
import shutil
import requests

load_dotenv()

db_user = os.getenv("DB_USER")
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")
db_password = os.getenv("DB_PASSWORD")
redis_host = os.getenv("REDIS_HOST")
redis_port = os.getenv("REDIS_PORT")
redis_password = os.getenv("REDIS_PASSWORD")
bucket_name = os.getenv("AWS_BUCKET_NAME")
web_host = os.getenv("WEB_HOST")
web_port = os.getenv("WEB_PORT")

# DB 연결
conn = psycopg2.connect(host=db_host, database=db_name, user=db_user, password=db_password)

# Celery 인스턴스 생성
celery = Celery(
    'tasks',
    broker=f'redis://:{redis_password}@{redis_host}:{redis_port}/1',
    backend=f'redis://:{redis_password}@{redis_host}:{redis_port}/1'
)

# S3 클라이언트 생성
s3 = boto3.client('s3')

RESULT_FOLDER = "results"

def wav_to_mp3(wav_path):
    mp3_path = os.path.splitext(wav_path)[0]+'.mp3'
    
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3")
    
    return mp3_path

def musicprocess(output_dir, file_path):
    # 노래 보컬 분리
    novocals_path = os.path.join(output_dir, "no_vocals.wav")
    vocals_path = os.path.join(output_dir, "vocals.wav")
    if not os.path.isfile(novocals_path) and not os.path.isfile(vocals_path):
        separate_audio(file_path, output_dir)

    # 보컬 전처리
    preprocessed_path = os.path.join(output_dir, "vocals_preprocessed.wav")
    if not os.path.isfile(preprocessed_path):
        vocal_preprocess(output_dir)

    # 보컬에서 가사 추출
    lyrics_path = os.path.join(output_dir, "lyrics.json")
    if not os.path.isfile(lyrics_path):
        lyrics = transcribe_audio(output_dir)
    else:
        with open(lyrics_path, 'r') as f:
            lyrics = json.load(f)

    # 보컬에서 음정 정보 추출
    pitch_paths = [os.path.join(output_dir, file) for file in ("frequency.npy", "confidence.npy", "activation.npy")]
    if not all(os.path.isfile(path) for path in pitch_paths):
        frequency, confidence, activation = pitch_extract(output_dir, lyrics)
    else:
        frequency, confidence, activation = (np.load(path) for path in pitch_paths)

    return lyrics, frequency, confidence, pitch_paths[2]

@celery.task(bind=True, autoretry_for=(Exception,), retry_backoff=5, retry_kwargs={'max_retries': 5})
def process(self, songId, file_name):
    # 작업 ID로 디렉토리 생성
    requestId = self.request.id
    output_dir = os.path.join(RESULT_FOLDER, requestId)
    os.makedirs(output_dir, exist_ok=True)

    # 처리할 파일 다운로드
    try:
        file_path = os.path.join(output_dir, os.path.basename(file_name))
        if not os.path.isfile(file_path):
            s3.download_file(bucket_name, file_name, file_path)
            file_name = os.path.basename(file_name) # 혹시 몰라서
    except:
        return {"status": "error", "msg": "cannot download file", "traceback": traceback.format_exc()}

    # # 노래 이미지 다운로드
    # try:
    #     img_path = os.path.join(output_dir, img_name)
    #     if not os.path.isfile(img_path):
    #         s3.download_file(bucket_name, img_name, img_path)
    # except:
    #     img_path = None

    lyrics, frequency, confidence, activation_path = musicprocess(output_dir, file_path)

    # # 원본노래 파일 
    # with open(file_path, 'rb') as f:
    #     original = f.read()
    # inst. 파일 
    no_vocals_path = wav_to_mp3(os.path.join(output_dir, "no_vocals.wav"))
    no_vocals_key = os.path.join(requestId, os.path.basename(no_vocals_path))
    s3.upload_file(no_vocals_path, bucket_name, no_vocals_key)
    # vocal 파일
    vocals_path = wav_to_mp3(os.path.join(output_dir, "vocals.wav"))
    vocals_key = os.path.join(requestId, os.path.basename(vocals_path))
    s3.upload_file(vocals_path, bucket_name, vocals_key)
    # # 이미지 파일
    # if img_path is not None:
    #     with open(img_path, 'rb') as f:
    #         image_file = f.read()

    # 음정 정보 파일
    pitch = frequency.tolist()
    confidence = confidence.tolist()
    activation_key = os.path.join(requestId, os.path.basename(activation_path))
    s3.upload_file(activation_path, bucket_name, activation_key)

    # SQL
    # SQL 삽입 쿼리
    query = """
        UPDATE songs
        SET mr_key = %s, vocal_key = %s, pitch = %s, pitch_confidence = %s, pitch_activation = %s, lyrics = %s, status = 'success'
        WHERE id = %s;
    """

    for i in range(5):
        try:
            cur = conn.cursor()
            # 데이터 삽입 (BLOB 데이터는 psycopg2.Binary로 감싸서 처리)
            cur.execute(query, (no_vocals_key, vocals_key, pitch, confidence, activation_key, json.dumps(lyrics), songId))
        except (psycopg2.OperationalError, psycopg2.DatabaseError) as e:
            if i < 4:
                time.sleep(10)
                conn.rollback()
                continue
            else:
                raise e
        else:
            conn.commit()
            cur.close()
            try:
                response = requests.post(f"https://{web_host}/api/songs/completion-notify", json={"songId": songId, "requestId": requestId})
                request_result = "done (" + str(response.status_code) + ")"
            except Exception as e:
                request_result = "failed"
                print(e)
            shutil.rmtree(output_dir)
            return "process success with notify " + request_result