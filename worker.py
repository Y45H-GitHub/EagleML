import os
import math
import json
import requests
import numpy as np
from PIL import Image
from io import BytesIO

print("🔧 Starting worker...")

# --- Load environment variables ---
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env loaded (if local)")
except Exception as e:
    print("⚠️ Could not load .env:", e)

# --- Import modules ---
try:
    import pika
    from skin_analysis import analyze_image_all_regions
    print("✅ All modules imported successfully")
except Exception as e:
    print("❌ Import error:", e)
    raise

# --- Config from environment ---
try:
    SCAN_QUEUE = os.environ.get("SCAN_QUEUE", "scan.queue")
    SCAN_EXCHANGE = os.environ.get("SCAN_EXCHANGE", "scan.exchange")
    SCAN_ROUTING_KEY = os.environ.get("SCAN_ROUTING_KEY", "scan.request")
    ML_RESULT_ENDPOINT = os.environ["ML_RESULT_ENDPOINT"]
    RABBITMQ_URL = os.environ["RABBITMQ_URL"]
    print("✅ Environment variables loaded")
except KeyError as e:
    print(f"❌ Missing required environment variable: {e}")
    raise

# --- RabbitMQ Setup ---
try:
    print("🔌 Connecting to RabbitMQ...")
    params = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.exchange_declare(exchange=SCAN_EXCHANGE, exchange_type='direct', durable=True)
    channel.queue_declare(queue=SCAN_QUEUE, durable=True)
    channel.queue_bind(exchange=SCAN_EXCHANGE, queue=SCAN_QUEUE, routing_key=SCAN_ROUTING_KEY)
    print("✅ Connected to RabbitMQ")
except Exception as e:
    print("❌ Failed to connect to RabbitMQ:", e)
    raise

# --- Helper: clean analysis result for safe JSON serialization ---
def clean_analysis_result(data):
    if isinstance(data, dict):
        return {k: clean_analysis_result(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_analysis_result(v) for v in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    else:
        return data

# --- Helper: Download and process image ---
def process_image_from_url(url):
    try:
        print(f"📷 Downloading image from {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return np.array(image)
    except Exception as e:
        print("❌ Failed to download/process image:", e)
        raise

# --- Callback ---
def callback(ch, method, properties, body):
    try:
        print("📨 Received message:", body)

        message = json.loads(json.loads(body))
        scan_session_id = message['scanSessionId']
        image_url = message['imageUrl']
        print(f"🧪 Scan ID: {scan_session_id}")
        print(f"🌐 Image URL: {image_url}")

        image_np = process_image_from_url(image_url)
        analysis_result = analyze_image_all_regions(image_np)
        print("✅ Image analysis complete")

        # Compute overall glow index (filter out NaN/inf)
        glow_index_map = analysis_result.get("glow_index", {})
        glow_values = [v for v in glow_index_map.values()
                       if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)]
        overall_glow_index = round(sum(glow_values) / len(glow_values), 2) if glow_values else 0.0

        # Sanitize analysis data
        sanitized_result = clean_analysis_result(analysis_result)

        # Prepare payload
        payload = {
            "scanSessionId": scan_session_id,
            "overallGlowIndex": overall_glow_index,
            "analysisMetadata": json.dumps(sanitized_result, allow_nan=False),
            "regionMetrics": [],
            "visualOutputs": []  # Reserved for future use
        }

        for metric_type, region_map in sanitized_result.items():
            if isinstance(region_map, dict):
                for region_name, value in region_map.items():
                    if isinstance(value, (int, float)) and not math.isnan(value) and not math.isinf(value):
                        payload["regionMetrics"].append({
                            "regionName": region_name,
                            "metricType": metric_type,
                            "metricValue": float(value)
                        })

        print("📤 Sending result to backend...")
        response = requests.post(ML_RESULT_ENDPOINT, json=payload)
        print(f"✅ Backend response status: {response.status_code}")

        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print("❌ Error processing message:", e)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

# --- Start Worker ---
try:
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=SCAN_QUEUE, on_message_callback=callback)
    print("✅ Worker is ready. Waiting for scan requests...")
    channel.start_consuming()
except Exception as e:
    print("❌ Fatal error during startup or consumption:", e)
    raise
