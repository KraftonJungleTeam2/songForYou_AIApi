import requests
import uuid

# 서버 URL
url = 'http://localhost:5000/seperate'

# 파일 경로 설정 (예: 전송할 오디오 파일 경로)
file_path = 'love wins all cliped.wav'

# 파일을 전송하는 POST 요청
with open(file_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files, data={"key": str(uuid.uuid4()),
                                                     "user_id": 21,
                                                     "metadata": {'null': 0},
                                                     "is_public": True})
    
# 서버 응답 출력 (파일이 반환될 경우 파일로 저장)
if response.status_code == 200:
    print(response.text)
else:
    print(f"Error: {response.status_code}, {response.text}")