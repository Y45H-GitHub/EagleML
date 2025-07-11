import pika
import json
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from skin_analysis import analyze_image_all_regions

# Constants matching Spring Boot config
SCAN_QUEUE = "scan.queue"
SCAN_EXCHANGE = "scan.exchange"
SCAN_ROUTING_KEY = "scan.request"
ML_RESULT_ENDPOINT = "https://eagle-backend-v1-production-734c.up.railway.app/api/scans/scan-analysis/complete"  # Local backend URL

# Use your full CloudAMQP URL
RABBITMQ_URL = "amqps://mwtpycti:uEcnEU2iCSJR7qq6Fce_LOBz-3-kHk7u@chimpanzee.rmq.cloudamqp.com/mwtpycti"

# Connect to RabbitMQ
params = pika.URLParameters(RABBITMQ_URL)
connection = pika.BlockingConnection(params)
channel = connection.channel()

# Declare exchange, queue, and bind them
channel.exchange_declare(exchange=SCAN_EXCHANGE, exchange_type='direct', durable=True)
channel.queue_declare(queue=SCAN_QUEUE, durable=True)
channel.queue_bind(exchange=SCAN_EXCHANGE, queue=SCAN_QUEUE, routing_key=SCAN_ROUTING_KEY)

def process_image_from_url(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return np.array(image)

def callback(ch, method, properties, body):
    try:
        print("Received message:", body)

        # Decode message (JSON string inside JSON string)
        message = json.loads(json.loads(body))
        scan_session_id = message['scanSessionId']
        image_url = message['imageUrl']

        image_np = process_image_from_url(image_url)
        analysis_result = analyze_image_all_regions(image_np)

        # Compute overallGlowIndex as average of glow_index values
        glow_index_map = analysis_result.get("glow_index", {})
        glow_values = list(glow_index_map.values())
        overall_glow_index = round(sum(glow_values) / len(glow_values), 2) if glow_values else 0.0

        # Prepare POST body for Spring Boot backend
        payload = {
            "scanSessionId": scan_session_id,
            "overallGlowIndex": overall_glow_index,
            "analysisMetadata": json.dumps(analysis_result),
            "regionMetrics": [],
            "visualOutputs": []  # Add later if needed
        }

        # Flatten region metrics
        for metric_type, region_map in analysis_result.items():
            if isinstance(region_map, dict):
                for region_name, value in region_map.items():
                    payload["regionMetrics"].append({
                        "regionName": region_name,
                        "metricType": metric_type,
                        "metricValue": float(value)
                    })

        print("Sending payload to backend:", json.dumps(payload, indent=2))
        response = requests.post(ML_RESULT_ENDPOINT, json=payload)
        print(f"Result sent to backend: {response.status_code}")

        ch.basic_ack(delivery_tag=method.delivery_tag)

    except Exception as e:
        print("Error processing message:", e)
        ch.basic_nack(delivery_tag=method.delivery_tag)

channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue=SCAN_QUEUE, on_message_callback=callback)

print(' [*] Waiting for scan requests. To exit press CTRL+C')
channel.start_consuming()
