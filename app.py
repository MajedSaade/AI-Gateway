import asyncio
import base64
import datetime as dt
import io
import logging
import os
from typing import Optional
import uuid

import boto3
import httpx
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from dotenv import load_dotenv
from tinydb import TinyDB
from ultralytics import YOLO

load_dotenv()

app = FastAPI(title="ChatOps AI Gateway")
YOLO_MODEL = YOLO("yolov8n.pt")
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


class CommandPayload(BaseModel):
    user_id: str
    command_text: str
    image_url: Optional[str] = None


def _parse_command(command_text: str) -> tuple[str, str]:
    text = command_text.strip()
    if not text:
        return "", ""

    known_prefixes = {"ask", "analyze", "detect"}

    # Accept formats like: ask/..., /ask ..., ask ...
    normalized = text[1:] if text.startswith("/") else text

    if "/" in normalized:
        prefix, rest = normalized.split("/", 1)
        prefix = prefix.strip().lower()
        if prefix in known_prefixes:
            return prefix, rest.strip()

    if " " in normalized:
        prefix, rest = normalized.split(None, 1)
        prefix = prefix.strip().lower()
        if prefix in known_prefixes:
            return prefix, rest.strip()

    prefix = normalized.strip().lower()
    if prefix in known_prefixes:
        return prefix, ""

    return "", text


def _storage_backend() -> str:
    return os.getenv("STORAGE_BACKEND", "s3").strip().lower()


def _nosql_db_path() -> str:
    return os.getenv("NOSQL_DB_PATH", "local_test_store.json")


def _ollama_base_url() -> str:
    # Prefer OLLAMA_URL for container-to-container routing in Compose/Kubernetes.
    base_url = os.getenv("OLLAMA_URL") or os.getenv("OLLAMA_BASE_URL", "http://ollama-engine:11434")
    return base_url.strip()


def _ollama_generate_path() -> str:
    path = os.getenv("OLLAMA_GENERATE_PATH", "/api/generate").strip()
    return path if path.startswith("/") else f"/{path}"


def _ollama_generate_url() -> str:
    return f"{_ollama_base_url().rstrip('/')}{_ollama_generate_path()}"


def _ollama_model_ask() -> str:
    return os.getenv("OLLAMA_MODEL_ASK", "llama3.2:1b").strip()


def _ollama_model_analyze() -> str:
    return os.getenv("OLLAMA_MODEL_ANALYZE", "llava").strip()


def _ollama_timeout_seconds() -> float:
    timeout_str = os.getenv("OLLAMA_TIMEOUT_SECONDS", "60").strip()
    try:
        return float(timeout_str)
    except ValueError:
        return 60.0


def _get_bucket_name() -> str:
    bucket = os.getenv("S3_BUCKET_NAME")
    if not bucket:
        raise RuntimeError("S3_BUCKET_NAME is not set")
    return bucket


async def save_prompt_to_s3(user_id: str, prompt: str) -> str:
    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    key = f"prompts/{user_id}_{timestamp}.txt"
    s3_client = boto3.client("s3")

    try:
        await asyncio.to_thread(
            s3_client.put_object,
            Bucket=_get_bucket_name(),
            Key=key,
            Body=prompt.encode("utf-8"),
            ContentType="text/plain",
        )
    except (BotoCoreError, ClientError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail="Failed to upload prompt to S3") from exc

    return key


def _save_to_nosql(record: dict) -> str:
    db = TinyDB(_nosql_db_path())
    try:
        doc_id = db.insert(record)
        return str(doc_id)
    finally:
        db.close()


async def save_prompt(user_id: str, prompt: str) -> str:
    if _storage_backend() == "nosql":
        record_id = await asyncio.to_thread(
            _save_to_nosql,
            {
                "id": str(uuid.uuid4()),
                "kind": "prompt",
                "user_id": user_id,
                "prompt": prompt,
                "timestamp": dt.datetime.utcnow().isoformat() + "Z",
            },
        )
        return f"nosql/prompts/{record_id}"

    return await save_prompt_to_s3(user_id, prompt)


async def download_image(image_url: str) -> bytes:
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(image_url)
            response.raise_for_status()
            return response.content
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=500, detail="Failed to download image") from exc


async def save_image(user_id: str, image_bytes: bytes, prefix: str) -> str:
    if _storage_backend() == "nosql":
        record_id = await asyncio.to_thread(
            _save_to_nosql,
            {
                "id": str(uuid.uuid4()),
                "kind": "image",
                "prefix": prefix,
                "user_id": user_id,
                "content_type": "image/jpeg",
                "image_b64": base64.b64encode(image_bytes).decode("utf-8"),
                "timestamp": dt.datetime.utcnow().isoformat() + "Z",
            },
        )
        return f"nosql/{prefix}/{record_id}"

    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    key = f"{prefix}/{user_id}_{timestamp}.jpg"

    s3_client = boto3.client("s3")
    try:
        await asyncio.to_thread(
            s3_client.put_object,
            Bucket=_get_bucket_name(),
            Key=key,
            Body=image_bytes,
            ContentType="image/jpeg",
        )
    except (BotoCoreError, ClientError, RuntimeError) as exc:
        raise HTTPException(status_code=500, detail="Failed to upload image to S3") from exc

    return key


async def download_image_and_upload_to_s3(
    user_id: str,
    image_url: str,
    prefix: str,
) -> tuple[str, bytes]:
    image_bytes = await download_image(image_url)
    key = await save_image(user_id=user_id, image_bytes=image_bytes, prefix=prefix)
    return key, image_bytes


def _extract_upstream_error(response: httpx.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict):
            for key in ("error", "message", "detail"):
                value = data.get(key)
                if value:
                    return str(value)[:240]
        if data:
            return str(data)[:240]
    except ValueError:
        pass

    text = response.text.strip().replace("\n", " ")
    return (text[:240] if text else "empty response body")


async def call_ollama_generate(payload: dict, command_prefix: str) -> str:
    generate_url = _ollama_generate_url()
    model_name = str(payload.get("model", ""))
    logger.info(
        "Ollama route prefix=%s outbound_url=%s model=%s",
        command_prefix,
        generate_url,
        model_name,
    )

    try:
        async with httpx.AsyncClient(timeout=_ollama_timeout_seconds()) as client:
            response = await client.post(generate_url, json=payload)

        logger.info(
            "Ollama response prefix=%s outbound_url=%s model=%s status=%s",
            command_prefix,
            generate_url,
            model_name,
            response.status_code,
        )

        if response.status_code != 200:
            upstream_error = _extract_upstream_error(response)
            logger.error(
                "Ollama failure prefix=%s outbound_url=%s model=%s status=%s error=%s",
                command_prefix,
                generate_url,
                model_name,
                response.status_code,
                upstream_error,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Ollama error ({response.status_code}): {upstream_error}",
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise HTTPException(status_code=502, detail="Ollama returned invalid JSON") from exc

        return str(data.get("response", ""))
    except httpx.TimeoutException as exc:
        logger.error(
            "Ollama unavailable prefix=%s outbound_url=%s model=%s reason=timeout",
            command_prefix,
            generate_url,
            model_name,
        )
        raise HTTPException(status_code=503, detail="Ollama unavailable: request timed out") from exc
    except httpx.RequestError as exc:
        logger.error(
            "Ollama unavailable prefix=%s outbound_url=%s model=%s reason=%s",
            command_prefix,
            generate_url,
            model_name,
            exc.__class__.__name__,
        )
        raise HTTPException(status_code=503, detail="Ollama unavailable: failed to connect") from exc


def _run_yolo_detection(image_bytes: bytes) -> list[dict]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = YOLO_MODEL(image, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "class_id": class_id,
                    "label": result.names.get(class_id, str(class_id)),
                    "confidence": confidence,
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    },
                }
            )
    return detections


@app.post("/process-command")
async def process_command(payload: CommandPayload):
    command_prefix, raw_prompt = _parse_command(payload.command_text)

    if command_prefix == "ask":
        if not raw_prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")

        storage_key: Optional[str] = None
        storage_error: Optional[str] = None
        try:
            storage_key = await save_prompt(payload.user_id, raw_prompt)
        except HTTPException as exc:
            storage_error = str(exc.detail)
            logger.warning(
                "ask storage write failed user_id=%s reason=%s",
                payload.user_id,
                storage_error,
            )

        response_text = await call_ollama_generate(
            {
                "model": _ollama_model_ask(),
                "prompt": raw_prompt,
                "stream": False,
            },
            command_prefix="ask",
        )
        return {
            "response_text": response_text,
            "storage_key": storage_key,
            "storage_error": storage_error,
            "storage_backend": _storage_backend(),
        }

    if command_prefix == "analyze":
        if not raw_prompt:
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        if not payload.image_url:
            raise HTTPException(status_code=400, detail="image_url is required for analyze/")

        storage_key, image_bytes = await download_image_and_upload_to_s3(
            user_id=payload.user_id,
            image_url=payload.image_url,
            prefix="analyzations",
        )
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        response_text = await call_ollama_generate(
            {
                "model": _ollama_model_analyze(),
                "prompt": raw_prompt,
                "stream": False,
                "images": [image_b64],
            },
            command_prefix="analyze",
        )
        return {
            "response_text": response_text,
            "storage_key": storage_key,
            "storage_backend": _storage_backend(),
        }

    if command_prefix == "detect":
        if not payload.image_url:
            raise HTTPException(status_code=400, detail="image_url is required for detect/")

        storage_key, image_bytes = await download_image_and_upload_to_s3(
            user_id=payload.user_id,
            image_url=payload.image_url,
            prefix="detections",
        )
        detections = await asyncio.to_thread(_run_yolo_detection, image_bytes)

        return {
            "status": "success",
            "model": "yolov8n.pt",
            "detections": detections,
            "count": len(detections),
            "storage_key": storage_key,
            "storage_backend": _storage_backend(),
        }

    raise HTTPException(
        status_code=400,
        detail="Unsupported command prefix. Use ask, analyze, or detect.",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
