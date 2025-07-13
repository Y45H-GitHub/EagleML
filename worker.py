import os
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
    import json
    import requests
    import numpy as np
    from PIL import Image
    from io import BytesIO
    from skin_analysis import analyze_image_all_regions
    print("✅ All modules imported successfully")
except Exception as e:
    print("❌ Import error:", e)
    raise

# --- Config from environment variables ---
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

# --- Image Processing ---
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

        # Parse nested JSON (string inside string)
        message = json.loads(json.loads(body))
        scan_session_id = message['scanSessionId']
        image_url = message['imageUrl']
        print(f"🧪 Scan ID: {scan_session_id}")
        print(f"🌐 Image URL: {image_url}")

        image_np = process_image_from_url(image_url)
        analysis_result = analyze_image_all_regions(image_np)
        print("✅ Image analysis complete")

        # Compute overall glow index
        glow_index_map = analysis_result.get("glow_index", {})
        glow_values = list(glow_index_map.values())
        overall_glow_index = round(sum(glow_values) / len(glow_values), 2) if glow_values else 0.0

        # Prepare payload
        payload = {
            "scanSessionId": scan_session_id,
            "overallGlowIndex": overall_glow_index,
            "analysisMetadata": json.dumps(analysis_result),
            "regionMetrics": [],
            "visualOutputs": []
        }

        for metric_type, region_map in analysis_result.items():
            if isinstance(region_map, dict):
                for region_item in region_map.items():
                    try:
                        region_name, value = region_item
                        payload["regionMetrics"].append({
                            "regionName": region_name,
                            "metricType": metric_type,
                            "metricValue": float(value)
                        })
                    except Exception as e:
                        print(f"⚠️ Skipping malformed region item in {metric_type}: {region_item} - {e}")
            else:
                print(f"⚠️ Skipping non-dict metric: {metric_type} = {region_map}")

        # Send to backend
        print("📤 Sending result to backend...")
        response = requests.post(ML_RESULT_ENDPOINT, json=payload)
        print(f"✅ Backend response status: {response.status_code}")

        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print("❌ Error processing message:", e)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)  # Avoid infinite retry loop

# --- Start worker ---
try:
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=SCAN_QUEUE, on_message_callback=callback)
    print("✅ Worker is ready. Waiting for scan requests...")
    channel.start_consuming()
except Exception as e:
    print("❌ Fatal error during startup or consumption:", e)
    raise
