# demucs==4.0.1
import os
import shutil
from demucs.separate import main as demucs_main

# Demucs를 호출하여 오디오 분리
def separate_audio(file_path, output_dir):
    # Demucs 호출 (필요한 인자를 전달)
    demucs_args = [
        file_path,
        '-o', output_dir,  # 출력 디렉토리 지정
        '--two-stems', 'vocals'  # 보컬만 추출 (필요에 따라 변경 가능)
    ]
    demucs_main(demucs_args)
    
    filename = os.path.splitext(os.path.basename(file_path))[0]
    shutil.move(os.path.join(output_dir, 'htdemucs', filename, 'vocals.wav'), output_dir)
    shutil.move(os.path.join(output_dir, 'htdemucs', filename, 'no_vocals.wav'), output_dir)

    return output_dir
