from celery import Celery
from dotenv import load_dotenv
import os

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

app = Celery('client', broker=f'redis://:{redis_password}@{redis_host}:{redis_port}/1')
# app = Celery('client', broker=f'redis://localhost:6379/0')

# 문자열로 작업 함수 호출
result = app.send_task('tasks.process', task_id='12345', args=[12, "songs/25/1729938723669-yesterday-remastered-2009-128-ytshorts.savetube.me.mp3"])
# result = app.send_task('tasks.long_task', task_id='1', args=[5])
print("Task ID:", result.id)


a={
    "body": "W1sxMiwgInNvbmdzLzI1LzE3Mjk5Mzg3MjM2NjkteWVzdGVyZGF5LXJlbWFzdGVyZWQtMjAwOS0xMjgteXRzaG9ydHMuc2F2ZXR1YmUubWUubXAzIl0sIHt9LCB7ImNhbGxiYWNrcyI6IG51bGwsICJlcnJiYWNrcyI6IG51bGwsICJjaGFpbiI6IG51bGwsICJjaG9yZCI6IG51bGx9XQ==", 
    "content-encoding": "utf-8", 
    "content-type": "application/json", 
    "headers": {
        "lang": "py", 
        "task": "tasks.process", 
        "id": "123",
        "shadow": null,
        "eta": null,
        "expires": null,
        "group": null,
        "group_index": null,
        "retries": 0,
        "timelimit": [null, null],
        "root_id": "123",
        "parent_id": null,
        "argsrepr": "[12, 'songs/25/1729938723669-yesterday-remastered-2009-128-ytshorts.savetube.me.mp3']",
        "kwargsrepr": "{}",
        "origin": "gen110988@DESKTOP-S7VR1T3",
        "ignore_result": false,
        "replaced_task_nesting": 0,
        "stamped_headers": null,
        "stamps": {}}, 
    "properties": {
        "correlation_id": "123",
        "reply_to": "aca5b4f9-111b-3078-9629-0a89f354ce27",
        "delivery_mode": 2,
        "delivery_info": {"exchange": "", "routing_key": "celery"},
        "priority": 0,
        "body_encoding": "base64",
        "delivery_tag": "947ad3a6-3abe-4fb3-b61d-fa55da0d6efe"
        }
 }
a={
    "body": "W1sxMiwgInNvbmdzLzI1LzE3Mjk5Mzg3MjM2NjkteWVzdGVyZGF5LXJlbWFzdGVyZWQtMjAwOS0xMjgteXRzaG9ydHMuc2F2ZXR1YmUubWUubXAzIl0sIHt9LCB7ImNhbGxiYWNrcyI6IG51bGwsICJlcnJiYWNrcyI6IG51bGwsICJjaGFpbiI6IG51bGwsICJjaG9yZCI6IG51bGx9XQ==", 
    "content-encoding": "utf-8", 
    "content-type": "application/json", 
    "headers": {
        "task": "tasks.process", 
        "id": "1234556",
        "retries": 0,}, 
    "properties": {
        "delivery_info": {"exchange": "", "routing_key": "celery"},
        "body_encoding": "base64",
        "delivery_tag": uuid
        }
 }
b={
    "body": {
        "id":"f87c86fd-1b73-475c-b36a-32984dc621a2",
        "task":"tasks.process",
        "args":[19,"songs/25/1729941962674-yesterday-remastered-2009-128-ytshorts.savetube.me.mp3"],
        "kwargs":{},
        "retries":0,
        "eta":null,
        "expires":null
        },
    "content-type":"application/json",
    "content-encoding":"utf-8",
    "headers":{}
    }