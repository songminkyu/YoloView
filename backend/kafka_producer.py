"""
프로덕션 레벨 Kafka Producer - FastAPI Backend
YoloView Detection 결과를 Kafka로 전송
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from kafka import KafkaProducer
from kafka.errors import KafkaError
import json
import uuid
import logging
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
import os
from pathlib import Path
import aiofiles
import hashlib

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경 변수
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_TOPIC_REQUEST = os.getenv('KAFKA_TOPIC_REQUEST', 'detection.request')
KAFKA_TOPIC_RESULT = os.getenv('KAFKA_TOPIC_RESULT', 'detection.result')
UPLOAD_DIR = Path(os.getenv('UPLOAD_DIR', './uploads'))
UPLOAD_DIR.mkdir(exist_ok=True)

# Pydantic 모델
class DetectionRequest(BaseModel):
    model_name: str = Field(default="yolov8n", description="YOLO 모델 이름")
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    img_size: int = Field(default=640, ge=320, le=1280)
    classes: Optional[List[int]] = Field(default=None, description="특정 클래스만 탐지")

class DetectionResponse(BaseModel):
    job_id: str
    status: str
    message: str
    file_hash: str
    created_at: str

class DetectionStatus(BaseModel):
    job_id: str
    status: str
    progress: Optional[int] = None
    result_count: Optional[int] = None
    error: Optional[str] = None

# Kafka Producer 싱글톤
class KafkaProducerSingleton:
    _instance = None
    _producer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def initialize(self):
        """Producer 초기화"""
        try:
            self._producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(','),
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',  # 모든 replica에서 확인
                retries=3,  # 실패 시 재시도
                max_in_flight_requests_per_connection=5,
                compression_type='gzip',  # 압축
                linger_ms=10,  # 배치 처리를 위한 대기 시간
                batch_size=16384,
                buffer_memory=33554432,  # 32MB
                request_timeout_ms=30000,
                metadata_max_age_ms=300000,
            )
            logger.info(f"Kafka Producer initialized: {KAFKA_BOOTSTRAP_SERVERS}")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka Producer: {e}")
            raise
    
    def get_producer(self) -> KafkaProducer:
        """Producer 인스턴스 반환"""
        if self._producer is None:
            self.initialize()
        return self._producer
    
    def close(self):
        """Producer 종료"""
        if self._producer:
            self._producer.flush()
            self._producer.close()
            logger.info("Kafka Producer closed")

# FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up application...")
    producer_singleton = KafkaProducerSingleton()
    producer_singleton.initialize()
    yield
    # Shutdown
    logger.info("Shutting down application...")
    producer_singleton.close()

# FastAPI 앱
app = FastAPI(
    title="YOLO Detection Service",
    description="YoloView Detection 결과를 Kafka로 전송하는 API",
    version="1.0.0",
    lifespan=lifespan
)

# 유틸리티 함수
async def calculate_file_hash(file_path: Path) -> str:
    """파일 해시 계산 (중복 체크용)"""
    hash_md5 = hashlib.md5()
    async with aiofiles.open(file_path, 'rb') as f:
        while chunk := await f.read(8192):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

async def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    """업로드 파일 저장"""
    async with aiofiles.open(destination, 'wb') as out_file:
        while content := await upload_file.read(1024 * 1024):  # 1MB chunks
            await out_file.write(content)
    return destination

def send_to_kafka_sync(topic: str, key: str, value: dict, producer: KafkaProducer) -> bool:
    """Kafka로 메시지 전송 (동기)"""
    try:
        future = producer.send(
            topic=topic,
            key=key,
            value=value,
            partition=None  # 자동 파티셔닝
        )
        
        # 전송 결과 대기 (타임아웃 10초)
        record_metadata = future.get(timeout=10)
        
        logger.info(
            f"Message sent to {record_metadata.topic} "
            f"partition {record_metadata.partition} "
            f"offset {record_metadata.offset}"
        )
        return True
        
    except KafkaError as e:
        logger.error(f"Failed to send message to Kafka: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending message: {e}")
        return False

# API 엔드포인트
@app.get("/")
async def root():
    """헬스 체크"""
    return {
        "service": "YOLO Detection Service",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """상세 헬스 체크"""
    try:
        producer = KafkaProducerSingleton().get_producer()
        # Kafka 연결 확인
        producer.bootstrap_connected()
        kafka_status = "connected"
    except Exception as e:
        kafka_status = f"disconnected: {str(e)}"
    
    return {
        "status": "healthy" if kafka_status == "connected" else "unhealthy",
        "kafka": kafka_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/detect", response_model=DetectionResponse)
async def create_detection_job(
    file: UploadFile = File(..., description="이미지 또는 비디오 파일"),
    model_name: str = "yolov8n",
    confidence: float = 0.25,
    iou_threshold: float = 0.45,
    img_size: int = 640,
    classes: Optional[str] = None  # "0,1,2" 형식
):
    """
    Detection 작업 생성 및 Kafka로 전송
    """
    # Job ID 생성
    job_id = str(uuid.uuid4())
    timestamp = datetime.now()
    
    # 파일 검증
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.mp4', '.avi', '.mov'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )
    
    try:
        # 파일 저장
        file_path = UPLOAD_DIR / f"{job_id}{file_ext}"
        await save_upload_file(file, file_path)
        
        # 파일 해시 계산
        file_hash = await calculate_file_hash(file_path)
        
        # classes 파싱
        classes_list = None
        if classes:
            try:
                classes_list = [int(c.strip()) for c in classes.split(',')]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid classes format")
        
        # Kafka 메시지 구성
        message = {
            "job_id": job_id,
            "file_path": str(file_path),
            "file_name": file.filename,
            "file_hash": file_hash,
            "file_size": file_path.stat().st_size,
            "file_type": file_ext,
            "model_name": model_name,
            "confidence": confidence,
            "iou_threshold": iou_threshold,
            "img_size": img_size,
            "classes": classes_list,
            "status": "pending",
            "created_at": timestamp.isoformat(),
            "priority": "normal"
        }
        
        # Kafka로 전송
        producer = KafkaProducerSingleton().get_producer()
        success = send_to_kafka_sync(
            topic=KAFKA_TOPIC_REQUEST,
            key=job_id,
            value=message,
            producer=producer
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to send message to Kafka"
            )
        
        logger.info(f"Detection job created: {job_id}")
        
        return DetectionResponse(
            job_id=job_id,
            status="pending",
            message="Detection job created successfully",
            file_hash=file_hash,
            created_at=timestamp.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating detection job: {e}")
        # 실패 시 파일 삭제
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/batch")
async def create_batch_detection_job(
    files: List[UploadFile] = File(..., description="여러 이미지 파일"),
    model_name: str = "yolov8n",
    confidence: float = 0.25,
):
    """
    배치 Detection 작업 생성
    """
    batch_id = str(uuid.uuid4())
    job_ids = []
    
    for file in files:
        result = await create_detection_job(
            file=file,
            model_name=model_name,
            confidence=confidence
        )
        job_ids.append(result.job_id)
    
    return {
        "batch_id": batch_id,
        "job_ids": job_ids,
        "total_jobs": len(job_ids),
        "status": "processing"
    }

@app.get("/jobs/{job_id}/status", response_model=DetectionStatus)
async def get_job_status(job_id: str):
    """
    작업 상태 조회 (실제로는 DB나 Redis에서 조회해야 함)
    """
    # TODO: Redis나 DB에서 실제 상태 조회
    return DetectionStatus(
        job_id=job_id,
        status="processing",
        progress=50,
        result_count=None
    )

@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    작업 취소
    """
    # TODO: Redis나 DB에 취소 상태 업데이트
    return {"message": f"Job {job_id} cancelled"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
