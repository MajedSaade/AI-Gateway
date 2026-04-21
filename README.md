# ChatOps AI Gateway

This project is a FastAPI gateway that accepts chat-style commands and routes them to AI backends.

## Features

- `ask/` command: sends text prompt to Ollama (model from env, default `llama3.2:1b`).
- `analyze/` command: downloads an image URL, stores it, and sends image + prompt to Ollama (model from env, default `llava`).
- `detect/` command: downloads and stores image, then runs local YOLO detection (`yolov8n.pt`).

## Requirements

- Python 3.10+
- Running Ollama instance (URL/path configurable via env)
- AWS credentials configured in environment or host
- `S3_BUCKET_NAME` environment variable

## Environment Variables

Gateway routes all command execution via `POST /process-command`, and internally dispatches by command prefix:

- `ask/...` -> Ollama text generation
- `analyze/...` -> Ollama multimodal analysis
- `detect/...` -> YOLO detection

Ollama config defaults:

```env
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_GENERATE_PATH=/api/generate
OLLAMA_MODEL_ASK=llama3.2:1b
OLLAMA_MODEL_ANALYZE=llava
OLLAMA_TIMEOUT_SECONDS=60
```

## Local Testing Without S3

For local tests, you can switch storage from S3 to a local NoSQL file database.

Set these environment variables in your local `.env`:

```env
STORAGE_BACKEND=nosql
NOSQL_DB_PATH=local_test_store.json
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_GENERATE_PATH=/api/generate
OLLAMA_MODEL_ASK=llama3.2:1b
OLLAMA_MODEL_ANALYZE=llava
OLLAMA_TIMEOUT_SECONDS=60
```

Behavior in local NoSQL mode:

- Prompts and downloaded images are saved into `local_test_store.json` (TinyDB).
- API responses include `storage_key` and `storage_backend` fields.
- No S3 write is performed.

## Setup

1. Create and activate virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run service:

```bash
python app.py
```

Service URL: `http://localhost:8080`

## API

### POST `/process-command`

Request body:

```json
{
	"user_id": "user-1",
	"command_text": "ask/ Explain FastAPI briefly",
	"image_url": "https://example.com/image.jpg"
}
```

Notes:

- `image_url` is required for `analyze/` and `detect/`.
- `image_url` is ignored for `ask/`.

## Deployment

Deployment uses Docker Compose on Linux hosts.

Host prerequisites:

- Docker installed and running
- Docker Compose available (`docker compose` or `docker-compose`)

The GitHub workflow uses Docker Hub build + reusable deploy workflow:

- [.github/workflows/build.yaml](.github/workflows/build.yaml)
- [.github/workflows/deploy.yaml](.github/workflows/deploy.yaml)

The deploy workflow copies [docker-compose.app.yaml](docker-compose.app.yaml) to EC2 and starts it with Docker Compose.

### Environment Template

Use [.env.example](.env.example) as your reference and adapt values for local or production deployment.

### GitHub Actions Setup

This repository includes:

- `.github/workflows/ci.yaml` for syntax checks on push and PR.
- `.github/workflows/deploy.yaml` for deployment on push to `main`.

Add these GitHub repository secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
- `EC2_HOST`
- `EC2_USERNAME`
- `EC2_SSH_KEY`
- `S3_BUCKET_NAME` (required when `STORAGE_BACKEND=s3`)

Static deployment values are currently hardcoded in workflows:

- `DOCKERHUB_REPOSITORY=ai-gateway`
- `ENVIRONMENT=production`
- `STORAGE_BACKEND=s3`
- `NOSQL_DB_PATH=local_test_store.json`
- `AWS_DEFAULT_REGION=eu-north-1`
- `OLLAMA_URL=http://host.docker.internal:11434`
- `OLLAMA_GENERATE_PATH=/api/generate`
- `OLLAMA_MODEL_ASK=llama3.2:1b`
- `OLLAMA_MODEL_ANALYZE=llava`
- `OLLAMA_TIMEOUT_SECONDS=60`

### Deployment Flow

1. Push code to `main`.
2. `build.yaml` builds and pushes image tag `${{ github.run_number }}` to Docker Hub.
3. `build.yaml` calls `deploy.yaml` with the image tag.
4. `deploy.yaml` copies compose file to EC2, creates `.env`, pulls image, and runs Docker Compose.