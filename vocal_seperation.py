# demucs==4.0.1
import os
import shutil
from demucs.separate import main as demucs_main
import uuid

# Demucs를 호출하여 오디오 분리
def separate_audio(file_path):
    # 고유 ID로 결과 디렉토리 생성
    output_dir = os.path.join(RESULT_FOLDER, str(uuid.uuid4()))
    os.makedirs(output_dir, exist_ok=True)

    # Demucs 호출 (필요한 인자를 전달)
    demucs_args = [
        file_path,
        '-o', output_dir,  # 출력 디렉토리 지정
        '--two-stems', 'vocals'  # 보컬만 추출 (필요에 따라 변경 가능)
    ]
    demucs_main(demucs_args)

    return output_dir
