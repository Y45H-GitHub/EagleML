import os
import math
import json
import orjson
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import pika
import time
import signal
import sys
from skin_analysis import analyze_image_all_regions

# --- Helpers ---
def clean_analysis_result(data):
    if isinstance(data, dict):
        return {k: clean_analysis_result(v) for k, v in data.items()}
    if isinstance(data, list):
        return [clean_analysis_result(v) for v in data]
    if isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
        return None
    return data

def process_image_from_url(url):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return np.array(img)

def report_failure(body):
    try:
        message = orjson.loads(body)
        sid = message.get("scanSessionId") or message.get("sessionId")
        failure_payload = {
            "scanSessionId": sid,
            "status": "failed",
            "error": "max_retries_exceeded"
        }
        requests.post(ML_RESULT_ENDPOINT, json=failure_payload, timeout=5)
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

def callback(ch, method, properties, body):
    global last_msg_time
    last_msg_time = time.time()
    start_time = time.perf_counter()
    sid = None
    try:
        msg = orjson.loads(body)
        sid = msg.get("scanSessionId") or msg.get("sessionId")
        url = msg.get("imageUrl")
        is_test = properties.headers.get("x-test", False)
        is_anon = properties.headers.get("x-anon", False)

        print(f"📨 Received scanSessionId={sid} | test={is_test} | anon={is_anon}")

        arr = process_image_from_url(url)
        print("✅ Image downloaded")

        result = analyze_image_all_regions(arr)
        print("✅ ML analysis complete")

        clean = clean_analysis_result(result)
        glow_vals = [v for v in clean.get("glow_index", {}).values() if isinstance(v, (int, float))]
        overall = round(sum(glow_vals) / len(glow_vals), 2) if glow_vals else 0.0

        payload = {
            ("sessionId" if is_anon else "scanSessionId"): sid,
            "overallGlowIndex": overall,
            "analysisMetadata": json.dumps(clean, allow_nan=False),
            "regionMetrics": [],
            "visualOutputs": [],
            "status": "completed"
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

        endpoint = ANON_RESULT_ENDPOINT if is_anon else ML_RESULT_ENDPOINT
        resp = requests.post(endpoint, json=payload, timeout=10)
        print(f"✅ Sent result to {endpoint} for {sid}, status {resp.status_code}")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print(f"❌ Error processing {sid if sid else 'message'}: {e}")
        republish_with_retry(body, properties, ch, method.delivery_tag)
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"⏱️ Total time for {sid if sid else 'message'}: {elapsed:.3f}s")

def connect():
    params = pika.URLParameters(RABBITMQ_URL)
    params.heartbeat = 20
    params.blocked_connection_timeout = 300
    conn = pika.BlockingConnection(params)
    ch = conn.channel()
    ch.exchange_declare(exchange=SCAN_EXCHANGE, exchange_type="direct", durable=True)
    ch.queue_declare(queue=SCAN_QUEUE, durable=True)
    ch.queue_bind(exchange=SCAN_EXCHANGE, queue=SCAN_QUEUE, routing_key=SCAN_ROUTING_KEY)
    return conn, ch

# --- MAIN ---
if __name__ == "__main__":
    print("🔧 Starting worker…")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ .env loaded (if local)")
    except ImportError:
        pass

    SCAN_QUEUE        = os.getenv("SCAN_QUEUE", "scan.queue")
    SCAN_EXCHANGE     = os.getenv("SCAN_EXCHANGE", "scan.exchange")
    SCAN_ROUTING_KEY  = os.getenv("SCAN_ROUTING_KEY", "scan.request")
    ML_RESULT_ENDPOINT= os.getenv("ML_RESULT_ENDPOINT")
    ANON_RESULT_ENDPOINT= os.getenv("ANON_RESULT_ENDPOINT")
    RABBITMQ_URL      = os.environ["RABBITMQ_URL"]
    MAX_IDLE          = float(os.getenv("MAX_IDLE", "1200"))
    PREFETCH_COUNT    = int(os.getenv("PREFETCH", "1"))

    print("RabbitMQ URL:", RABBITMQ_URL)
    connection, channel = connect()
    channel.basic_qos(prefetch_count=PREFETCH_COUNT)
    print(f"✅ Connected to RabbitMQ (queue: {SCAN_QUEUE}, exchange: {SCAN_EXCHANGE})")

    # initialize last_msg_time so idle timer starts now
    last_msg_time = time.time()

    # consume setup
    channel.basic_consume(queue=SCAN_QUEUE, on_message_callback=callback)
    print("✅ Worker ready; waiting for messages…")

    try:
        # instead of start_consuming, loop and periodically check idle
        while True:
            connection.process_data_events(time_limit=1)  # wait up to 1s for events
            if time.time() - last_msg_time > MAX_IDLE:
                print(f"🛌 Idle for {int(time.time() - last_msg_time)}s; shutting down.")
                break
    except KeyboardInterrupt:
        print("👋 Shutdown requested")
    finally:
        if connection.is_open:
            channel.stop_consuming()
            connection.close()
        print("👋 Worker shut down.")