import os
import math
import json
import orjson
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import pika
from skin_analysis import analyze_image_all_regions

print("🔧 Starting worker...")

# --- Load .env locally ---
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env loaded (if local)")
except ImportError:
    pass

# --- Config from environment ---
SCAN_QUEUE         = os.getenv("SCAN_QUEUE", "scan.queue")
SCAN_EXCHANGE      = os.getenv("SCAN_EXCHANGE", "scan.exchange")
SCAN_ROUTING_KEY   = os.getenv("SCAN_ROUTING_KEY", "scan.request")
ML_RESULT_ENDPOINT = os.environ["ML_RESULT_ENDPOINT"]
RABBITMQ_URL       = os.environ["RABBITMQ_URL"]

# --- Persistent HTTP session ---
http = requests.Session()
http.headers.update({"User-Agent": "skin-worker/1.0"})

# --- RabbitMQ connection helper ---
def connect():
    params = pika.URLParameters(RABBITMQ_URL)
    params.heartbeat = 600
    params.blocked_connection_timeout = 300
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.exchange_declare(exchange=SCAN_EXCHANGE, exchange_type="direct", durable=True)
    ch.queue_declare(queue=SCAN_QUEUE, durable=True)
    ch.queue_bind(exchange=SCAN_EXCHANGE, queue=SCAN_QUEUE, routing_key=SCAN_ROUTING_KEY)
    ch.basic_qos(prefetch_count=1)
    return conn, ch

connection, channel = connect()
print("✅ Connected to RabbitMQ")

# --- Helpers ---
def clean_analysis_result(data):
    if isinstance(data, dict):
        return {k: clean_analysis_result(v) for k, v in data.items()}
    if isinstance(data, list):
        return [clean_analysis_result(v) for v in data]
    if isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
    return data

def process_image_from_url(url):
    resp = http.get(url, timeout=10)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return np.array(img)

def report_failure(body):
    try:
        message = orjson.loads(orjson.loads(body))
        sid = message.get("scanSessionId")
        failure_payload = {
            "scanSessionId": sid,
            "status": "failed",
            "error": "max_retries_exceeded"
        }
        http.post(ML_RESULT_ENDPOINT, json=failure_payload, timeout=5)
    except Exception as ex:
        print(f"⚠️ Could not report failure: {ex}")

def republish_with_retry(body, props, ch, delivery_tag):
    headers = dict(props.headers or {})
    attempts = headers.get("x-attempts", 0) + 1
    if attempts <= 3:
        headers["x-attempts"] = attempts
        ch.basic_publish(
            exchange=SCAN_EXCHANGE,
            routing_key=SCAN_ROUTING_KEY,
            body=body,
            properties=pika.BasicProperties(
                content_type=props.content_type,
                headers=headers,
                delivery_mode=2
            )
        )
        print(f"♻️  Requeued (attempt {attempts}/3)")
    else:
        print("⚠️  Max retries reached; dropping message")
        report_failure(body)
    ch.basic_ack(delivery_tag=delivery_tag)

import time  # Add this at the top of your file if not already present

def callback(ch, method, properties, body):
    start_time = time.perf_counter()
    try:
        message = orjson.loads(orjson.loads(body))
        sid = message.get("scanSessionId")
        url = message.get("imageUrl")
        print(f"📨 Received scanSessionId={sid}")

        start_time1 = time.perf_counter()
        arr = process_image_from_url(url) 
        end_time1 = time.perf_counter()
        total_sec1 = end_time1 - start_time1
        print(f"⏱️ Total cloudinary img time  {sid}: {total_sec1:.3f} seconds")

        result = analyze_image_all_regions(arr) 


        glow_map = result.get("glow_index", {})
        glow_vals = [v for v in glow_map.values() if isinstance(v, (int, float))]
        overall = round(sum(glow_vals) / len(glow_vals), 2) if glow_vals else 0.0

        clean = clean_analysis_result(result)
        payload = {
            "scanSessionId": sid,
            "overallGlowIndex": overall,
            "analysisMetadata": json.dumps(clean, allow_nan=False),
            "regionMetrics": [],
            "visualOutputs": []
        }
        for metric, regions in clean.items():
            if isinstance(regions, dict):
                for name, val in regions.items():
                    if isinstance(val, (int, float)):
                        payload["regionMetrics"].append({
                            "regionName": name,
                            "metricType": metric,
                            "metricValue": float(val)
                        })

        resp = http.post(ML_RESULT_ENDPOINT, json=payload, timeout=10)
        print(f"✅ Sent result for {sid} with status {resp.status_code}")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print(f"❌ Error processing message: {e}")
        republish_with_retry(body, properties, ch, method.delivery_tag)
    finally:
        end_time = time.perf_counter()
        total_sec = end_time - start_time
        print(f"⏱️ Total time taken for {sid}: {total_sec:.3f} seconds")


# --- Start consuming ---
channel.basic_consume(queue=SCAN_QUEUE, on_message_callback=callback)
print("✅ Worker is ready. Waiting for scan requests...")
try:
    channel.start_consuming()
except KeyboardInterrupt:
    print("👋 Shutdown requested")
finally:
    channel.stop_consuming()
    connection.close()
