import os
import json
import time
import logging
import logstash
from kafka import KafkaConsumer
import psycopg2
import redis

# --- Configuration ---
# If running locally, use localhost. If in docker, these will be overridden by env vars.
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9093")
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:password@localhost:5432/yolodb")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
LOGSTASH_HOST = os.getenv("LOGSTASH_HOST", "localhost")
LOGSTASH_PORT = int(os.getenv("LOGSTASH_PORT", 5000))

# --- Logging ---
logger = logging.getLogger('python-logstash-logger')
logger.setLevel(logging.INFO)
logger.addHandler(logstash.TCPLogstashHandler(LOGSTASH_HOST, LOGSTASH_PORT, version=1))

# --- Clients ---
redis_client = redis.from_url(REDIS_URL)

def get_db_connection():
    try:
        conn = psycopg2.connect(POSTGRES_URL)
        return conn
    except Exception as e:
        logger.error(f"Postgres connection failed: {e}")
        return None

def init_db():
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(100),
                    detections JSONB,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source VARCHAR(255)
                )
            """)
        conn.commit()
        conn.close()
        logger.info("Database initialized")

def process_message(message):
    data = json.loads(message.value.decode('utf-8'))
    logger.info(f"Processing detection from {data.get('source')}")
    
    # Store in Postgres
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO detections (model_name, detections, source) VALUES (%s, %s, %s)",
                    (data['model_name'], json.dumps(data['detections']), data['source'])
                )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save to Postgres: {e}")

    # Update Redis Stats
    try:
        redis_client.incr("total_detections")
        redis_client.hincrby("model_stats", data['model_name'], 1)
    except Exception as e:
        logger.error(f"Failed to update Redis: {e}")

def main():
    # Wait for DB to be ready
    db_retries = 10

    while db_retries > 0:
        if get_db_connection():
            init_db()
            break
        db_retries -= 1
        logger.warning(f"Waiting for Postgres ({db_retries} retries left)...")
        time.sleep(5)

    # Wait for Kafka to be ready
    kafka_retries = 10
    consumer = None
    while kafka_retries > 0:
        try:
            consumer = KafkaConsumer(
                "yolo-detections",
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                group_id="yolo-group",
                auto_offset_reset='earliest'
            )
            logger.info("Consumer connected to Kafka")
            break
        except Exception as e:
            kafka_retries -= 1
            logger.warning(f"Waiting for Kafka ({kafka_retries} retries left): {e}")
            time.sleep(5)
    
    if not consumer:
        logger.error("Consumer failed to connect to Kafka after multiple retries")
        return

    logger.info("Consumer started, waiting for detections...")
    for message in consumer:
        process_message(message)

if __name__ == "__main__":
    main()
