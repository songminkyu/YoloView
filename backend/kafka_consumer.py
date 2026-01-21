"""
프로덕션 레벨 Kafka Consumer - Detection Worker
YoloView 모델을 실행하고 결과를 DB에 저장
"""

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
import json
import logging
import signal
import sys
import asyncio
import asyncpg
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import os
from dataclasses import dataclass
import traceback
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import cv2
import numpy as np
from ultralytics import YOLO

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('consumer.log')
    ]
)
logger = logging.getLogger(__name__)

# 환경 변수
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_TOPIC_REQUEST = os.getenv('KAFKA_TOPIC_REQUEST', 'detection.request')
KAFKA_TOPIC_RESULT = os.getenv('KAFKA_TOPIC_RESULT', 'detection.result')
KAFKA_TOPIC_ERROR = os.getenv('KAFKA_TOPIC_ERROR', 'detection.error')
KAFKA_GROUP_ID = os.getenv('KAFKA_GROUP_ID', 'detection-consumer-group')
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/detection_db')
MODEL_DIR = Path(os.getenv('MODEL_DIR', './models'))
RESULT_DIR = Path(os.getenv('RESULT_DIR', './results'))
RESULT_DIR.mkdir(exist_ok=True)

# Prometheus 메트릭
messages_processed = Counter('messages_processed_total', 'Total messages processed', ['status'])
processing_time = Histogram('message_processing_seconds', 'Time to process message')
active_jobs = Gauge('active_jobs', 'Number of jobs currently processing')
detection_count = Counter('detections_total', 'Total detections by class', ['class_name'])
model_inference_time = Histogram('model_inference_seconds', 'Model inference time')

@dataclass
class DetectionResult:
    """Detection 결과 데이터 클래스"""
    job_id: str
    frame_number: int
    class_id: int
    class_name: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    bbox_width: float
    bbox_height: float
    timestamp: datetime

class DatabaseManager:
    """PostgreSQL 데이터베이스 관리"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """DB 연결 풀 생성"""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=60,
                max_queries=50000,
                max_inactive_connection_lifetime=300
            )
            logger.info("Database connection pool created")
            
            # 테이블 생성
            await self._create_tables()
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    async def _create_tables(self):
        """필요한 테이블 생성"""
        async with self.pool.acquire() as conn:
            # 작업 테이블
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS detection_jobs (
                    id SERIAL PRIMARY KEY,
                    job_id UUID UNIQUE NOT NULL,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    model_name VARCHAR(50) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    total_frames INTEGER,
                    processed_frames INTEGER DEFAULT 0,
                    total_detections INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP
                )
            ''')
            
            # 결과 테이블
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS detection_results (
                    id SERIAL PRIMARY KEY,
                    job_id UUID NOT NULL REFERENCES detection_jobs(job_id) ON DELETE CASCADE,
                    frame_number INTEGER NOT NULL,
                    class_id INTEGER NOT NULL,
                    class_name VARCHAR(100) NOT NULL,
                    confidence REAL NOT NULL,
                    bbox_x1 REAL NOT NULL,
                    bbox_y1 REAL NOT NULL,
                    bbox_x2 REAL NOT NULL,
                    bbox_y2 REAL NOT NULL,
                    bbox_width REAL NOT NULL,
                    bbox_height REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            ''')
            
            # 인덱스 생성
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_detection_results_job_id 
                ON detection_results(job_id)
            ''')
            
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_detection_results_class_name 
                ON detection_results(class_name)
            ''')
            
            logger.info("Database tables created/verified")
    
    async def create_job(self, job_data: dict):
        """작업 생성"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO detection_jobs 
                (job_id, file_path, file_name, file_hash, model_name, status, started_at)
                VALUES ($1, $2, $3, $4, $5, $6, NOW())
            ''', job_data['job_id'], job_data['file_path'], 
                job_data['file_name'], job_data['file_hash'],
                job_data['model_name'], 'processing')
    
    async def update_job_status(self, job_id: str, status: str, 
                                error_message: Optional[str] = None):
        """작업 상태 업데이트"""
        async with self.pool.acquire() as conn:
            if status == 'completed':
                await conn.execute('''
                    UPDATE detection_jobs 
                    SET status = $1, completed_at = NOW()
                    WHERE job_id = $2
                ''', status, job_id)
            elif status == 'failed':
                await conn.execute('''
                    UPDATE detection_jobs 
                    SET status = $1, error_message = $2, completed_at = NOW()
                    WHERE job_id = $3
                ''', status, error_message, job_id)
    
    async def save_results(self, results: List[DetectionResult]):
        """Detection 결과 저장 (배치)"""
        if not results:
            return
        
        async with self.pool.acquire() as conn:
            await conn.executemany('''
                INSERT INTO detection_results 
                (job_id, frame_number, class_id, class_name, confidence,
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_width, bbox_height)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ''', [(r.job_id, r.frame_number, r.class_id, r.class_name, 
                   r.confidence, r.bbox_x1, r.bbox_y1, r.bbox_x2, r.bbox_y2,
                   r.bbox_width, r.bbox_height) for r in results])
            
            # 통계 업데이트
            job_id = results[0].job_id
            await conn.execute('''
                UPDATE detection_jobs 
                SET total_detections = total_detections + $1
                WHERE job_id = $2
            ''', len(results), job_id)
    
    async def close(self):
        """연결 풀 종료"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

class YOLODetector:
    """YOLO 모델 관리 및 추론"""
    
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.models: Dict[str, YOLO] = {}
    
    def load_model(self, model_name: str) -> YOLO:
        """모델 로드 (캐싱)"""
        if model_name in self.models:
            return self.models[model_name]
        
        model_path = self.model_dir / f"{model_name}.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading model: {model_name}")
        model = YOLO(str(model_path))
        self.models[model_name] = model
        
        return model
    
    @model_inference_time.time()
    def detect(self, image: np.ndarray, model: YOLO, 
               confidence: float, iou_threshold: float,
               img_size: int, classes: Optional[List[int]] = None) -> List[dict]:
        """이미지에서 객체 탐지"""
        results = model.predict(
            image,
            conf=confidence,
            iou=iou_threshold,
            imgsz=img_size,
            classes=classes,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                
                detections.append({
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'confidence': conf,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'width': float(x2 - x1),
                    'height': float(y2 - y1)
                })
                
                # Prometheus 메트릭
                detection_count.labels(class_name=cls_name).inc()
        
        return detections

class DetectionConsumer:
    """Kafka Consumer 메인 클래스"""
    
    def __init__(self):
        self.running = True
        self.consumer: Optional[KafkaConsumer] = None
        self.producer: Optional[KafkaProducer] = None
        self.db: Optional[DatabaseManager] = None
        self.detector: Optional[YOLODetector] = None
        
        # Graceful shutdown 설정
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """종료 시그널 처리"""
        logger.info(f"Received shutdown signal: {signum}")
        self.running = False
    
    def _init_kafka(self):
        """Kafka 초기화"""
        # Consumer 초기화
        self.consumer = KafkaConsumer(
            KAFKA_TOPIC_REQUEST,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(','),
            group_id=KAFKA_GROUP_ID,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='earliest',  # 처음부터 읽기
            enable_auto_commit=False,  # 수동 커밋
            max_poll_records=10,  # 한 번에 가져올 메시지 수
            max_poll_interval_ms=300000,  # 5분
            session_timeout_ms=30000,
            heartbeat_interval_ms=3000,
        )
        
        # Producer 초기화 (결과 전송용)
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(','),
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
            acks='all',
            retries=3,
        )
        
        logger.info("Kafka consumer and producer initialized")
    
    async def _init_resources(self):
        """리소스 초기화"""
        # Database 초기화
        self.db = DatabaseManager(DATABASE_URL)
        await self.db.connect()
        
        # YOLO Detector 초기화
        self.detector = YOLODetector(MODEL_DIR)
        
        logger.info("All resources initialized")
    
    @processing_time.time()
    async def _process_message(self, message: dict):
        """메시지 처리"""
        job_id = message['job_id']
        
        try:
            active_jobs.inc()
            logger.info(f"Processing job: {job_id}")
            
            # DB에 작업 생성
            await self.db.create_job(message)
            
            # 파일 로드
            file_path = Path(message['file_path'])
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # 모델 로드
            model = self.detector.load_model(message['model_name'])
            
            # 이미지/비디오 처리
            file_ext = file_path.suffix.lower()
            
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                # 이미지 처리
                results = await self._process_image(
                    file_path, model, message, job_id
                )
            elif file_ext in ['.mp4', '.avi', '.mov']:
                # 비디오 처리
                results = await self._process_video(
                    file_path, model, message, job_id
                )
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            # 결과를 Kafka로 전송
            self._send_result(job_id, results, 'completed')
            
            # DB 상태 업데이트
            await self.db.update_job_status(job_id, 'completed')
            
            messages_processed.labels(status='success').inc()
            logger.info(f"Job completed: {job_id}, detections: {len(results)}")
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            logger.error(traceback.format_exc())
            
            # 에러 전송
            self._send_error(job_id, str(e))
            
            # DB 상태 업데이트
            await self.db.update_job_status(job_id, 'failed', str(e))
            
            messages_processed.labels(status='error').inc()
        
        finally:
            active_jobs.dec()
    
    async def _process_image(self, file_path: Path, model: YOLO, 
                            message: dict, job_id: str) -> List[DetectionResult]:
        """이미지 처리"""
        image = cv2.imread(str(file_path))
        if image is None:
            raise ValueError(f"Failed to load image: {file_path}")
        
        detections = self.detector.detect(
            image, model,
            message['confidence'],
            message['iou_threshold'],
            message['img_size'],
            message.get('classes')
        )
        
        # DetectionResult 객체로 변환
        results = []
        for det in detections:
            result = DetectionResult(
                job_id=job_id,
                frame_number=0,
                class_id=det['class_id'],
                class_name=det['class_name'],
                confidence=det['confidence'],
                bbox_x1=det['bbox'][0],
                bbox_y1=det['bbox'][1],
                bbox_x2=det['bbox'][2],
                bbox_y2=det['bbox'][3],
                bbox_width=det['width'],
                bbox_height=det['height'],
                timestamp=datetime.now()
            )
            results.append(result)
        
        # DB 저장
        await self.db.save_results(results)
        
        return results
    
    async def _process_video(self, file_path: Path, model: YOLO,
                            message: dict, job_id: str) -> List[DetectionResult]:
        """비디오 처리"""
        cap = cv2.VideoCapture(str(file_path))
        frame_count = 0
        all_results = []
        
        try:
            while cap.isOpened() and self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                detections = self.detector.detect(
                    frame, model,
                    message['confidence'],
                    message['iou_threshold'],
                    message['img_size'],
                    message.get('classes')
                )
                
                # DetectionResult 객체로 변환
                for det in detections:
                    result = DetectionResult(
                        job_id=job_id,
                        frame_number=frame_count,
                        class_id=det['class_id'],
                        class_name=det['class_name'],
                        confidence=det['confidence'],
                        bbox_x1=det['bbox'][0],
                        bbox_y1=det['bbox'][1],
                        bbox_x2=det['bbox'][2],
                        bbox_y2=det['bbox'][3],
                        bbox_width=det['width'],
                        bbox_height=det['height'],
                        timestamp=datetime.now()
                    )
                    all_results.append(result)
                
                frame_count += 1
                
                # 배치 저장 (100 프레임마다)
                if frame_count % 100 == 0:
                    await self.db.save_results(all_results)
                    all_results = []
                    logger.info(f"Processed {frame_count} frames for job {job_id}")
            
            # 남은 결과 저장
            if all_results:
                await self.db.save_results(all_results)
            
        finally:
            cap.release()
        
        return all_results
    
    def _send_result(self, job_id: str, results: List[DetectionResult], status: str):
        """결과를 Kafka로 전송"""
        message = {
            'job_id': job_id,
            'status': status,
            'total_detections': len(results),
            'timestamp': datetime.now().isoformat()
        }
        
        self.producer.send(KAFKA_TOPIC_RESULT, value=message)
        self.producer.flush()
    
    def _send_error(self, job_id: str, error: str):
        """에러를 Kafka로 전송"""
        message = {
            'job_id': job_id,
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        
        self.producer.send(KAFKA_TOPIC_ERROR, value=message)
        self.producer.flush()
    
    async def run(self):
        """메인 실행 루프"""
        logger.info("Starting Detection Consumer...")
        
        # 리소스 초기화
        self._init_kafka()
        await self._init_resources()
        
        # Prometheus 메트릭 서버 시작
        start_http_server(8001)
        logger.info("Prometheus metrics server started on port 8001")
        
        logger.info(f"Listening to topic: {KAFKA_TOPIC_REQUEST}")
        
        try:
            while self.running:
                # Kafka에서 메시지 가져오기
                messages = self.consumer.poll(timeout_ms=1000)
                
                for topic_partition, records in messages.items():
                    for record in records:
                        try:
                            await self._process_message(record.value)
                            
                            # 수동 커밋
                            self.consumer.commit()
                            
                        except Exception as e:
                            logger.error(f"Error processing record: {e}")
                            # 다음 메시지 처리 계속
                            continue
                
                # CPU 부하 방지
                await asyncio.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Fatal error in consumer loop: {e}")
            logger.error(traceback.format_exc())
        
        finally:
            # 정리
            logger.info("Shutting down consumer...")
            if self.consumer:
                self.consumer.close()
            if self.producer:
                self.producer.close()
            if self.db:
                await self.db.close()
            logger.info("Consumer shutdown complete")

if __name__ == "__main__":
    consumer = DetectionConsumer()
    asyncio.run(consumer.run())
