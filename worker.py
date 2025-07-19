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

class Worker:
    """
    A resilient RabbitMQ worker that handles connection loss and graceful shutdowns.
    """
    def __init__(self):
        print("🔧 Initializing worker...")
        # Load configuration from environment variables
        self.scan_queue = os.getenv("SCAN_QUEUE", "scan.queue")
        self.scan_exchange = os.getenv("SCAN_EXCHANGE", "scan.exchange")
        self.scan_routing_key = os.getenv("SCAN_ROUTING_KEY", "scan.request")
        self.ml_result_endpoint = os.getenv("ML_RESULT_ENDPOINT")
        self.anon_result_endpoint = os.getenv("ANON_RESULT_ENDPOINT")
        self.rabbitmq_url = os.environ["RABBITMQ_URL"]
        self.prefetch_count = int(os.getenv("PREFETCH", "1"))

        self.connection = None
        self.channel = None
        self.is_shutting_down = False

    def _setup_signal_handlers(self):
        """Sets up handlers for shutdown signals."""
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Gracefully handles a shutdown signal."""
        if not self.is_shutting_down:
            print(f"\n👋 Shutdown signal ({signum}) received. Finishing current message...")
            self.is_shutting_down = True
            if self.channel:
                self.channel.stop_consuming()

    def _connect(self):
        """Establishes a connection and channel to RabbitMQ."""
        print("🔗 Connecting to RabbitMQ...")
        params = pika.URLParameters(self.rabbitmq_url)
        params.heartbeat = 30 # Increased heartbeat for better resilience
        params.blocked_connection_timeout = 300
        self.connection = pika.BlockingConnection(params)
        self.channel = self.connection.channel()
        self.channel.exchange_declare(exchange=self.scan_exchange, exchange_type="direct", durable=True)
        self.channel.queue_declare(queue=self.scan_queue, durable=True)
        self.channel.queue_bind(exchange=self.scan_exchange, queue=self.scan_queue, routing_key=self.scan_routing_key)
        self.channel.basic_qos(prefetch_count=self.prefetch_count)
        print(f"✅ Connected to RabbitMQ (queue: {self.scan_queue})")

    def _close_connection(self):
        """Closes the RabbitMQ connection and channel if they are open."""
        try:
            if self.channel and self.channel.is_open:
                self.channel.close()
            if self.connection and self.connection.is_open:
                self.connection.close()
            print("🚪 Connection closed.")
        except Exception as e:
            print(f"⚠️ Error closing connection: {e}")

    def run(self):
        """
        The main loop of the worker. Connects and consumes messages, with a
        reconnection mechanism in case of failure.
        """
        self._setup_signal_handlers()
        while not self.is_shutting_down:
            try:
                self._connect()
                print("✅ Worker ready; waiting for messages…")
                self.channel.basic_consume(queue=self.scan_queue, on_message_callback=self.callback)
                self.channel.start_consuming()
            except pika.exceptions.AMQPConnectionError as e:
                print(f"❌ Connection lost: {e}. Retrying in 5 seconds...")
                self._close_connection()
                time.sleep(5)
            except Exception as e:
                print(f"❌ An unexpected error occurred: {e}. Retrying in 10 seconds...")
                self._close_connection()
                time.sleep(10)
        
        self._close_connection()
        print("👋 Worker has shut down gracefully.")

    # --- Your original helper and callback functions are now methods of the class ---

    def clean_analysis_result(self, data):
        if isinstance(data, dict):
            return {k: self.clean_analysis_result(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self.clean_analysis_result(v) for v in data]
        if isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
            return None
        return data

    def process_image_from_url(self, url):
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        return np.array(img)

    def report_failure(self, body):
        try:
            message = orjson.loads(body)
            sid = message.get("scanSessionId") or message.get("sessionId")
            failure_payload = {"scanSessionId": sid, "status": "failed", "error": "max_retries_exceeded"}
            requests.post(self.ml_result_endpoint, json=failure_payload, timeout=5)
        except Exception as ex:
            print(f"⚠️ Could not report failure: {ex}")

    def republish_with_retry(self, body, props, ch, delivery_tag):
        headers = dict(props.headers or {})
        attempts = headers.get("x-attempts", 0) + 1
        if attempts <= 3:
            headers["x-attempts"] = attempts
            ch.basic_publish(
                exchange=self.scan_exchange,
                routing_key=self.scan_routing_key,
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
            self.report_failure(body)
        ch.basic_ack(delivery_tag=delivery_tag)

    def callback(self, ch, method, properties, body):
        start_time = time.perf_counter()
        sid = None
        try:
            msg = orjson.loads(body)
            sid = msg.get("scanSessionId") or msg.get("sessionId")
            url = msg.get("imageUrl")
            is_test = properties.headers.get("x-test", False)
            is_anon = properties.headers.get("x-anon", False)

            print(f"📨 Received scanSessionId={sid} | test={is_test} | anon={is_anon}")

            arr = self.process_image_from_url(url)
            print("✅ Image downloaded")

            result = analyze_image_all_regions(arr)
            print("✅ ML analysis complete")

            clean = self.clean_analysis_result(result)
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

            endpoint = self.anon_result_endpoint if is_anon else self.ml_result_endpoint
            resp = requests.post(endpoint, json=payload, timeout=10)
            print(f"✅ Sent result to {endpoint} for {sid}, status {resp.status_code}")
            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            print(f"❌ Error processing {sid if sid else 'message'}: {e}")
            self.republish_with_retry(body, properties, ch, method.delivery_tag)
        finally:
            elapsed = time.perf_counter() - start_time
            print(f"⏱️ Total time for {sid if sid else 'message'}: {elapsed:.3f}s")

# --- MAIN ---
if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ .env loaded (if local)")
    except ImportError:
        pass
    
    worker = Worker()
    worker.run()
