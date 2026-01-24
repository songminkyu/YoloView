import os
import json
import logging
import logstash
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from kafka import KafkaProducer
from prometheus_fastapi_instrumentator import Instrumentator
import asyncio
import uvicorn

from contextlib import asynccontextmanager

# --- Logging ---
LOGSTASH_HOST = os.getenv("LOGSTASH_HOST", "logstash")
LOGSTASH_PORT = int(os.getenv("LOGSTASH_PORT", 5000))
logger = logging.getLogger('python-logstash-logger')
logger.setLevel(logging.INFO)
logger.addHandler(logstash.TCPLogstashHandler(LOGSTASH_HOST, LOGSTASH_PORT, version=1))

# --- Kafka ---
# If running locally (not in docker), use localhost:9093 by default
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9093")
producer = None

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global producer
    # Use fewer retries and shorter sleep for local debugging if requested
    is_debug = os.getenv("DEBUG", "true").lower() == "true"
    retries = 3 if is_debug else 10
    retry_interval = 2 if is_debug else 5
    
    while retries > 0:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                request_timeout_ms=1000  # Fail fast during local debug
            )
            logger.info(f"API connected to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
            break
        except Exception as e:
            retries -= 1
            if retries > 0:
                logger.warning(f"Waiting for Kafka at {KAFKA_BOOTSTRAP_SERVERS} ({retries} retries left): {e}")
                await asyncio.sleep(retry_interval)
    
    if not producer:
        logger.error(f"API failed to connect to Kafka at {KAFKA_BOOTSTRAP_SERVERS}")
    
    yield
    
    # Shutdown logic
    if producer:
        producer.close()
        logger.info("Kafka producer closed")

app = FastAPI(title="YoloView API", lifespan=lifespan)

# --- Models ---
class DetectionResult(BaseModel):
    model_name: str
    detections: list
    timestamp: float
    source: str

# --- Endpoints ---
@app.post("/api/v1/detections")
async def receive_detections(result: DetectionResult):
    # FALLBACK: If producer is missing, just log for debugging instead of failing
    if not producer:
        logger.warning(f"[MOCK MODE] Kafka unavailable. Received detection: {result.model_name} from {result.source}")
        return {"status": "mock_accepted", "detail": "Kafka unavailable, logged locally"}
    
    try:
        producer.send("yolo-detections", value=result.dict())
        producer.flush()
        logger.info(f"Detection received and sent to Kafka: {result.model_name} from {result.source}")
        return {"status": "accepted"}
    except Exception as e:
        logger.error(f"Error sending to Kafka: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/health")
async def health():
    return {"status": "ok"}

# --- Monitoring ---
Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
