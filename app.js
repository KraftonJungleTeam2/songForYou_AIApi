const redis = require('redis');

const client = redis.createClient();
client.connect();

// Celery 작업 큐에 직접 작업 추가
// requestId는 songs테이블의 id(PK)이어야 함!
// filename은 S3에 업로드한 노래 이름
async function addTaskToCeleryQueue(filekey) {
    const requestId = (DB에 행을 생성한 뒤 나온 id)
    const task = {
        id: requestId,
        task: "tasks.process",  // Python의 Celery 작업 이름
        args: [filename],         // 작업에 전달할 인자
        kwargs: {},
        retries: 5, // 5회까지 재시작
        eta: null
    };

    // 작업을 JSON 문자열로 변환 후 Redis의 Celery 큐에 푸시
    await client.rPush("celery", JSON.stringify(task));

    console.log(`Task added to queue with ID: ${requestId}`);
    return requestId;
}

// 작업 ID로 상태 조회
async function task_status(req, res) {
    const taskId = req.params.task_id;
    const redisKey = `celery-task-meta-${taskId}`;

    // Redis에서 작업 상태 조회
    const taskData = await redisClient.get(redisKey);
    if (taskData) {
        const taskStatus = JSON.parse(taskData);
        res.json({
            task_id: taskId,
            status: taskStatus.status,
            result: taskStatus.result
        });
    }
}

// 사용 예제
(async () => {
    const filekey = "1/somesongs.mp3"
    const taskId = await addTaskToCeleryQueue(filekey);
    console.log(`Added task with ID: ${taskId}`);
})();
