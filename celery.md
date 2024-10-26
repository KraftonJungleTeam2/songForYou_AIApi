Celery 작업자를 실행하기 위한 명령어 규칙은 다음과 같습니다. 기본적인 명령어 형식은 `celery -A <app_name> worker`이며, 여기서 `<app_name>`은 Celery 인스턴스가 정의된 파일이나 모듈 이름을 가리킵니다. 필요에 따라 여러 옵션을 추가하여 작업자 설정을 세부적으로 조정할 수 있습니다.

### 1. 기본 명령어 형식

```bash
celery -A <app_name> worker
```

- **`-A <app_name>`**: Celery 인스턴스를 정의한 애플리케이션 모듈을 지정합니다. 예를 들어, `app.py`에 Celery 인스턴스가 정의되어 있다면 `-A app`을 사용합니다.
- **`worker`**: 작업자를 실행하라는 명령어입니다.

예시:
```bash
celery -A tasks worker --loglevel=info
```

여기서 `tasks`는 `tasks.py` 파일에 Celery 인스턴스가 정의되어 있는 경우를 가정한 것입니다.

### 2. 로그 레벨 설정

작업자 실행 시 로그 수준을 설정하여 작업자의 로그 출력을 조정할 수 있습니다.

```bash
celery -A <app_name> worker --loglevel=<loglevel>
```

- **`--loglevel`**: 로그 수준을 설정합니다. 주요 옵션은 `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`입니다. 일반적으로 `info` 또는 `debug` 수준이 많이 사용됩니다.

예시:
```bash
celery -A tasks worker --loglevel=info
```

### 3. 병렬 처리 (concurrency) 설정

작업자가 동시에 처리할 수 있는 작업 수를 지정하는 옵션입니다. CPU 코어 수에 맞춰 적절히 설정할 수 있습니다.

```bash
celery -A <app_name> worker --concurrency=<num_workers>
```

- **`--concurrency`**: 동시에 처리할 작업 수를 지정합니다. 기본값은 `cpu_count()`에 따라 자동으로 결정됩니다.

예시:
```bash
celery -A tasks worker --concurrency=4
```

### 4. 이름 지정 (hostname)

작업자 이름을 지정하여 여러 작업자를 구분할 수 있습니다.

```bash
celery -A <app_name> worker --hostname=<worker_name>
```

- **`--hostname`**: 특정 이름으로 작업자를 지정할 수 있습니다. 예를 들어 여러 작업자를 실행하는 경우 이름을 다르게 설정하여 구분할 수 있습니다.

예시:
```bash
celery -A tasks worker --hostname=worker1@%h
```

여기서 `%h`는 호스트 이름을 포함하여 작업자 이름을 자동으로 지정하는 데 유용한 템플릿 변수입니다.

### 5. 자동 재시작 (autoscale)

작업자의 최소 및 최대 작업 수를 지정하여 작업 부하에 따라 작업자 수를 자동으로 조정합니다.

```bash
celery -A <app_name> worker --autoscale=<max_workers>,<min_workers>
```

- **`--autoscale`**: 작업자의 최대 및 최소 수를 설정합니다. 예를 들어, `10,3`으로 설정하면 최소 3개에서 최대 10개의 작업자를 유지합니다.

예시:
```bash
celery -A tasks worker --autoscale=10,3
```

### 6. 데몬화 (Demonizing the Worker)

작업자를 백그라운드에서 실행하는 방법입니다.

```bash
celery multi start w1 -A <app_name> --logfile=/var/log/celery/%n%I.log --loglevel=info
```

이 명령어는 작업자를 백그라운드 데몬으로 실행하며, 로그 파일 경로를 지정할 수 있습니다.

### 요약

- 기본 실행: `celery -A <app_name> worker`
- 로그 레벨: `--loglevel=info`
- 동시 작업 수: `--concurrency=4`
- 작업자 이름: `--hostname=worker1@%h`
- 자동 조정: `--autoscale=10,3`

이 명령어들을 조합하여 Celery 작업자를 원하는 설정으로 실행할 수 있습니다.