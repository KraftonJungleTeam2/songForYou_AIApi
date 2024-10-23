import requests
import uuid

# 서버 URL
url = 'http://localhost:4000/separate'

# 파일 경로 설정 (예: 전송할 오디오 파일 경로)
file_path = 'Yesterday Remastered 2009.mp3'
img_path = 'img.jpg'

# 파일을 전송하는 POST 요청
with open(file_path, 'rb') as f, open(img_path, 'rb') as img:
    files = {'file': f, 'image': img}
    response = requests.post(url, files=files, data={"key": str(uuid.uuid4()),
                                                     "user_id": 23,
                                                     "metadata": "{'title': '가사 마스킹 테스트', 'description': '...'}",
                                                     "is_public": True,
                                                     "genre": "kpop"})

# 서버 응답 출력 (파일이 반환될 경우 파일로 저장)
if response.status_code == 200:
    print(response.text)
else:
    print(f"Error: {response.status_code}, {response.text}")