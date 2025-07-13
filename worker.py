import os
import pika
import json
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from skin_analysis import analyze_image_all_regions

# Load .env in local dev (won't affect Railway)
from dotenv import load_dotenv
load_dotenv()

# --- Config from environment variables ---
SCAN_QUEUE = os.environ.get("SCAN_QUEUE", "scan.queue")
SCAN_EXCHANGE = os.environ.get("SCAN_EXCHANGE", "scan.exchange")
SCAN_ROUTING_KEY = os.environ.get("SCAN_ROUTING_KEY", "scan.request")
ML_RESULT_ENDPOINT = os.environ["ML_RESULT_ENDPOINT"]  # Required in prod
RABBITMQ_URL = os.environ["RABBITMQ_URL"]              # Required in prod

# --- RabbitMQ Setup ---
params = pika.URLParameters(RABBITMQ_URL)
connection = pika.BlockingConnection(params)
channel = connection.channel()

channel.exchange_declare(exchange=SCAN_EXCHANGE, exchange_type='direct', durable=True)
channel.queue_declare(queue=SCAN_QUEUE, durable=True)
channel.queue_bind(exchange=SCAN_EXCHANGE, queue=SCAN_QUEUE, routing_key=SCAN_ROUTING_KEY)

# --- Image Processing ---
def process_image_from_url(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return np.array(image)

# --- Callback for Incoming Scan Messages ---
def callback(ch, method, properties, body):
    try:
        print("Received message:", body)

        # Parse nested JSON (string inside string)
        message = json.loads(json.loads(body))
        scan_session_id = message['scanSessionId']
        image_url = message['imageUrl']

        image_np = process_image_from_url(image_url)
        analysis_result = analyze_image_all_regions(image_np)

        # Compute overall glow index
        glow_index_map = analysis_result.get("glow_index", {})
        glow_values = list(glow_index_map.values())
        overall_glow_index = round(sum(glow_values) / len(glow_values), 2) if glow_values else 0.0

        # Build payload
        payload = {
            "scanSessionId": scan_session_id,
            "overallGlowIndex": overall_glow_index,
            "analysisMetadata": json.dumps(analysis_result),
            "regionMetrics": [],
            "visualOutputs": []
        }

        for metric_type, region_map in analysis_result.items():
            if isinstance(region_map, dict):
                for region_name, value in region_map.items():
                    payload["regionMetrics"].append({
                        "regionName": region_name,
                        "metricType": metric_type,
                        "metricValue": float(value)
                    })
            else:
                print(f"Skipping non-dict metric: {metric_type}")

        # Send result to Spring Boot backend
        print("Sending payload to backend...")
        response = requests.post(ML_RESULT_ENDPOINT, json=payload)
        print(f"Result status: {response.status_code}")

        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print("Error processing message:", e)
        ch.basic_nack(delivery_tag=method.delivery_tag)

# --- Start Worker ---
channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=SCAN_QUEUE, on_message_callback=callback)

print(" [*] Worker ready. Waiting for scan requests...")
channel.start_consuming()
