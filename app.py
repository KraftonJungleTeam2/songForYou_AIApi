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

app = Flask(__name__)

# 분리된 오디오 파일이 저장될 폴더 경로 설정
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# '.mp3', 'id' in
# save lyrics, inst., pitchdata
@app.route('/preprocess', methods=['POST'])
def upload_file():
    # 오디오 파일이 전달되었는지 확인
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    # 파일 저장
    if file:
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        # Demucs를 사용하여 오디오 분리
        output_dir = separate_audio(file_path)

        # 분리된 파일을 zip으로 묶기                                                                                                                                                                                                                                     
        zip_filename = f"{os.path.splitext(filename)[0]}_separated.zip"
        zip_filepath = os.path.join(RESULT_FOLDER, zip_filename)
        shutil.make_archive(zip_filepath.replace('.zip', ''), 'zip', output_dir)

        # zip 파일 반환
        return send_from_directory(RESULT_FOLDER, zip_filename, as_attachment=True)
    
    return jsonify({'error': 'File could not be processed'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
