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
from multiprocessing import Process, Queue, freeze_support
from skin_analysis import analyze_image_all_regions

# --- Helpers ---
# Recursively walks through your nested dict/lists, replacing any NaN or Inf float with None, so JSON serialization never breaks
def clean_analysis_result(data):
    if isinstance(data, dict):
        return {k: clean_analysis_result(v) for k, v in data.items()}
    if isinstance(data, list):
        return [clean_analysis_result(v) for v in data]
    if isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
        return None
    return data

# Downloads the JPEG/PNG from url via requests.get(), Converts it to a PIL image, then a NumPy RGB array, Raises on HTTP errors or timeouts.
def process_image_from_url(url):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    return np.array(img)

# If a message has permanently failed (3 retries), this: Parses the JSON body to get scanSessionId, Sends a “failed” POST to your ML_RESULT endpoint so your backend knows it errored out.
def report_failure(body):
    try:
        message = orjson.loads(body)
        sid = message.get("scanSessionId")
        failure_payload = {"scanSessionId": sid, "status": "failed", "error": "max_retries_exceeded"}
        requests.post(ML_RESULT_ENDPOINT, json=failure_payload, timeout=5)
    except Exception as ex:
        print(f"⚠️ Could not report failure: {ex}")

# Implements the “3‑retry” pattern:  Reads or sets header "x-attempts" 
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

def _ml_worker(arr, out_q):
    out_q.put(analyze_image_all_regions(arr))


# Record “last seen” time so you can auto‑exit if idle too long
# Start a timer to measure end‑to‑end latency
# Parse the JSON body for scanSessionId and imageUrl
# Download the image with process_image_from_url
# Fork a process to run analyze_image_all_regions(arr) (isolates your main worker if ML crashes)
# Join the process and retrieve the result dict
# Clean NaN/Inf with clean_analysis_result
# Compute overall glow index from the per‑region scores
# Build the POST payload, including regionMetrics array
# Send it to your backend (test vs prod endpoint based on header x-test)
# ACK the message on success, so RabbitMQ removes it
# On exception: log error, call republish_with_retry to handle retries
# Finally: print the total elapsed time and call check_idle_and_exit in case you’ve been idle too long

def callback(ch, method, properties, body):
    global last_msg_time
    last_msg_time = time.time()
    start_time = time.perf_counter()
    sid = None
    try:
        msg = orjson.loads(body)
        sid = msg.get("scanSessionId")
        url = msg.get("imageUrl")
        is_test = properties.headers.get("x-test", False)

        print(f"📨 Received scanSessionId={sid} | test={is_test}")

        arr = process_image_from_url(url)

        q = Queue()
        p = Process(target=_ml_worker, args=(arr, q))
        p.start()
        result = q.get()
        p.join()

        clean = clean_analysis_result(result)
        glow_vals = [v for v in clean.get("glow_index", {}).values() if isinstance(v, (int, float))]
        overall = round(sum(glow_vals) / len(glow_vals), 2) if glow_vals else 0.0

        payload = {
            "sessionId": sid,
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

        endpoint = ANON_RESULT_ENDPOINT if is_test else ML_RESULT_ENDPOINT
        resp = requests.post(endpoint, json=payload, timeout=10)
        print(f"✅ Sent result to {endpoint} for {sid}, status {resp.status_code}")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print(f"❌ Error processing {sid if sid else 'message'}: {e}")
        republish_with_retry(body, properties, ch, method.delivery_tag)
    finally:
        elapsed = time.perf_counter() - start_time
        print(f"⏱️ Total time for {sid if sid else 'message'}: {elapsed:.3f}s")
        check_idle_and_exit()

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

# If no message has arrived for more than MAX_IDLE seconds, it:
# Prints an “idle” message
# Closes RabbitMQ connection
# Calls sys.exit(0) to quit the process entirely—avoiding idle billing
def check_idle_and_exit():
    if time.time() - last_msg_time > MAX_IDLE:
        print(f"🛌 Idle for {int(time.time() - last_msg_time)}s; shutting down.")
        connection.close()
        sys.exit(0)

def shutdown_handler(signum, frame):
    print("🚫 Shutdown signal received; stopping consumer…")
    try:
        channel.stop_consuming()
    except Exception:
        pass
    finally:
        if connection.is_open:
            connection.close()
        sys.exit(0)

if __name__ == "__main__":
    freeze_support()  # Required for multiprocessing on Windows

    print("🔧 Starting worker…")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ .env loaded (if local)")
    except ImportError:
        pass

    # --- Config from environment ---
    SCAN_QUEUE = os.getenv("SCAN_QUEUE", "scan.queue")
    SCAN_EXCHANGE = os.getenv("SCAN_EXCHANGE", "scan.exchange")
    SCAN_ROUTING_KEY = os.getenv("SCAN_ROUTING_KEY", "scan.request")
    ML_RESULT_ENDPOINT = os.getenv("ML_RESULT_ENDPOINT")
    ANON_RESULT_ENDPOINT = os.getenv("ANON_RESULT_ENDPOINT", "https://eagle-backend-v1-production.up.railway.app/api/anonscans/analysis")
    RABBITMQ_URL = os.environ["RABBITMQ_URL"]
    MAX_IDLE = float(os.getenv("MAX_IDLE", "1200"))
    PREFETCH_COUNT = int(os.getenv("PREFETCH", "1"))
    print("RabbitMQ URL:", RABBITMQ_URL)

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    connection, channel = connect()
    channel.basic_qos(prefetch_count=PREFETCH_COUNT)
    print(f"✅ Connected to RabbitMQ (queue: {SCAN_QUEUE}, exchange: {SCAN_EXCHANGE}, routingKey: {SCAN_ROUTING_KEY})")

    channel.basic_consume(queue=SCAN_QUEUE, on_message_callback=callback)
    print("✅ Worker ready; waiting for messages…")

    try:
        channel.start_consuming()
    except Exception as e:
        print(f"⚠️ Consumer stopped: {e}")
    finally:
        if connection.is_open:
            connection.close()
        print("👋 Worker shut down.")