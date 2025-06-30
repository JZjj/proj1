import logging
import os
import json
import sys
import time
from typing import Dict, Any, Optional

from kafka import KafkaConsumer, KafkaProducer
import prometheus_client
from prometheus_client import Summary, Counter, Histogram
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("gpt4_nano_triton_consumer")

# Environment variables
KAFKA_BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP', 'localhost:9092')
TRITON_URL = os.getenv('TRITON_URL', 'http://localhost:8000')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt4_nano')

# Prometheus metrics
INFER_TIME = Summary('consumer_inference_seconds', 'Time spent in GPT-4.1 nano inference via Triton')
INFER_COUNT = Counter('consumer_inference_total', 'Total number of inference requests', ['status'])
INFER_TOKENS = Histogram('consumer_tokens_used', 'Number of tokens used per request')
ERROR_COUNT = Counter('consumer_errors_total', 'Total number of errors', ['error_type'])
RESPONSE_TIME = Histogram('consumer_response_time_seconds', 'Response time distribution')


class GPT4NanoTritonConsumer:
    """Kafka consumer for GPT-4.1 nano inference requests via Triton"""

    def __init__(self):
        """Initialize the consumer"""
        logger.info(f"Initializing GPT-4.1 nano Triton consumer...")
        logger.info(f"Kafka bootstrap server: {KAFKA_BOOTSTRAP}")
        logger.info(f"Triton URL: {TRITON_URL}")
        logger.info(f"Model name: {MODEL_NAME}")

        self.triton_url = TRITON_URL.rstrip('/')
        self.model_name = MODEL_NAME
        self.infer_url = f"{self.triton_url}/v2/models/{self.model_name}/infer"

        # Test Triton connection
        if not self._test_triton_health():
            raise RuntimeError("❌ Triton server is not accessible")

        # Initialize Kafka consumer
        self.consumer = KafkaConsumer(
            'inference-requests',
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_deserializer=lambda m: json.loads(m.decode()),
            auto_offset_reset='earliest',
            group_id='gpt4-nano-triton-consumer-group',
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000
        )

        # Initialize Kafka producer for results
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode(),
            retries=3,
            acks='all'
        )

        logger.info("✅ GPT-4.1 nano Triton consumer initialized successfully")

    def _test_triton_health(self) -> bool:
        """Test if Triton server is healthy and model is ready"""
        try:
            # Check server health
            health_url = f"{self.triton_url}/v2/health/ready"
            response = requests.get(health_url, timeout=5)

            if response.status_code != 200:
                logger.error(f"❌ Triton server not ready: {response.status_code}")
                return False

            # Check if model is ready
            model_url = f"{self.triton_url}/v2/models/{self.model_name}/ready"
            response = requests.get(model_url, timeout=5)

            if response.status_code == 200:
                logger.info(f"✅ Triton server and model '{self.model_name}' are ready")
                return True
            else:
                logger.error(f"❌ Model '{self.model_name}' not ready: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Triton health check failed: {e}")
            return False

    def _call_triton_inference(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call GPT-4.1 nano via Triton Inference Server

        Args:
            request_data: Request parameters

        Returns:
            Triton inference response
        """
        payload = request_data.get('payload', {})

        # Extract parameters with defaults
        prompt = payload.get('prompt', '')
        max_tokens = payload.get('max_tokens', 100)
        temperature = payload.get('temperature', 0.7)
        top_p = payload.get('top_p', 1.0)
        frequency_penalty = payload.get('frequency_penalty', 0.0)
        presence_penalty = payload.get('presence_penalty', 0.0)
        stop_sequences = payload.get('stop', None)

        # Validate required parameters
        if not prompt:
            raise ValueError("Prompt is required")

        # Prepare Triton inference payload
        triton_payload = {
            "inputs": [
                {
                    "name": "prompt",
                    "datatype": "BYTES",
                    "shape": [1],
                    "data": [prompt]
                },
                {
                    "name": "max_tokens",
                    "datatype": "INT32",
                    "shape": [1],
                    "data": [max_tokens]
                },
                {
                    "name": "temperature",
                    "datatype": "FP32",
                    "shape": [1],
                    "data": [temperature]
                },
                {
                    "name": "top_p",
                    "datatype": "FP32",
                    "shape": [1],
                    "data": [top_p]
                },
                {
                    "name": "frequency_penalty",
                    "datatype": "FP32",
                    "shape": [1],
                    "data": [frequency_penalty]
                },
                {
                    "name": "presence_penalty",
                    "datatype": "FP32",
                    "shape": [1],
                    "data": [presence_penalty]
                }
            ]
        }

        # Add stop sequences if provided
        if stop_sequences:
            if isinstance(stop_sequences, str):
                stop_sequences = [stop_sequences]

            triton_payload["inputs"].append({
                "name": "stop",
                "datatype": "BYTES",
                "shape": [len(stop_sequences)],
                "data": stop_sequences
            })

        logger.info(f"Calling Triton inference for prompt: {prompt[:100]}...")
        logger.debug(f"Triton payload: {json.dumps(triton_payload, indent=2)}")

        # Make the inference request
        response = requests.post(
            self.infer_url,
            json=triton_payload,
            timeout=120,  # 2 minutes timeout for GPT-4.1 nano
            headers={'Content-Type': 'application/json'}
        )

        response.raise_for_status()
        return response.json()

    def process_inference_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single inference request

        Args:
            request_data: The request data from Kafka message

        Returns:
            Dictionary with the inference result
        """
        start_time = time.time()
        request_id = request_data.get('request_id', 'unknown')

        try:
            logger.info(f"📨 Processing request {request_id}")

            # Call Triton inference
            with INFER_TIME.time():
                triton_response = self._call_triton_inference(request_data)

            processing_time = time.time() - start_time

            # Parse Triton response
            response_text = None
            tokens_used = None
            finish_reason = None
            model_version = None

            if 'outputs' in triton_response:
                for output in triton_response['outputs']:
                    output_name = output.get('name', '')
                    output_data = output.get('data', [])

                    if output_name == 'response_text' and output_data:
                        response_text = output_data[0]
                    elif output_name == 'tokens_used' and output_data:
                        tokens_used = output_data[0]
                    elif output_name == 'finish_reason' and output_data:
                        finish_reason = output_data[0]
                    elif output_name == 'model_version' and output_data:
                        model_version = output_data[0]

            # Prepare successful result
            result = {
                'request_id': request_id,
                'success': True,
                'output': {
                    'response_text': response_text,
                    'tokens_used': tokens_used,
                    'finish_reason': finish_reason,
                    'model': f"{self.model_name}:{model_version}" if model_version else self.model_name,
                    'processing_time': processing_time
                },
                'metadata': {
                    'triton_url': self.triton_url,
                    'model_name': self.model_name,
                    'timestamp': time.time(),
                    'raw_triton_response': triton_response
                }
            }

            # Update metrics
            INFER_COUNT.labels(status='success').inc()
            RESPONSE_TIME.observe(processing_time)

            if tokens_used:
                INFER_TOKENS.observe(tokens_used)

            logger.info(f"✅ Request {request_id} completed successfully")
            logger.info(f"📊 Tokens used: {tokens_used}")
            logger.info(f"⏱️  Processing time: {processing_time:.2f}s")
            logger.info(f"🔚 Finish reason: {finish_reason}")

            return result

        except ValueError as e:
            error_msg = f"Invalid request parameters: {e}"
            logger.error(f"❌ Request {request_id}: {error_msg}")

            ERROR_COUNT.labels(error_type='validation').inc()
            INFER_COUNT.labels(status='error').inc()

            return {
                'request_id': request_id,
                'success': False,
                'error': error_msg,
                'error_type': 'validation',
                'timestamp': time.time()
            }

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Triton connection failed: {e}"
            logger.error(f"❌ Request {request_id}: {error_msg}")

            ERROR_COUNT.labels(error_type='connection').inc()
            INFER_COUNT.labels(status='error').inc()

            return {
                'request_id': request_id,
                'success': False,
                'error': error_msg,
                'error_type': 'connection',
                'timestamp': time.time()
            }

        except requests.exceptions.Timeout as e:
            error_msg = f"Triton request timeout: {e}"
            logger.error(f"❌ Request {request_id}: {error_msg}")

            ERROR_COUNT.labels(error_type='timeout').inc()
            INFER_COUNT.labels(status='error').inc()

            return {
                'request_id': request_id,
                'success': False,
                'error': error_msg,
                'error_type': 'timeout',
                'timestamp': time.time()
            }

        except requests.exceptions.HTTPError as e:
            error_msg = f"Triton HTTP error: {e.response.status_code} - {e.response.text if e.response else 'No response'}"
            logger.error(f"❌ Request {request_id}: {error_msg}")

            ERROR_COUNT.labels(error_type='http').inc()
            INFER_COUNT.labels(status='error').inc()

            return {
                'request_id': request_id,
                'success': False,
                'error': error_msg,
                'error_type': 'http',
                'timestamp': time.time()
            }

        except Exception as e:
            error_msg = f"Unexpected error: {e}"
            logger.error(f"❌ Request {request_id}: {error_msg}")
            logger.exception("Full error traceback:")

            ERROR_COUNT.labels(error_type='unexpected').inc()
            INFER_COUNT.labels(status='error').inc()

            return {
                'request_id': request_id,
                'success': False,
                'error': error_msg,
                'error_type': 'unexpected',
                'timestamp': time.time()
            }

        finally:
            total_time = time.time() - start_time
            logger.info(f"Request {request_id} processed in {total_time:.2f}s")

    def send_result(self, result: Dict[str, Any]):
        """Send result back to Kafka"""
        try:
            future = self.producer.send('inference-results', value=result)
            # Wait for the send to complete with timeout
            record_metadata = future.get(timeout=10)

            logger.info(f"📤 Result sent for request {result['request_id']}")
            logger.debug(
                f"📍 Sent to topic: {record_metadata.topic}, partition: {record_metadata.partition}, offset: {record_metadata.offset}")

        except Exception as e:
            logger.error(f"❌ Failed to send result for request {result['request_id']}: {e}")
            ERROR_COUNT.labels(error_type='kafka_send').inc()

    def run(self):
        """Main consumer loop"""
        logger.info("🚀 Starting GPT-4.1 nano Triton consumer...")
        logger.info("⏳ Waiting for inference requests...")

        try:
            for message in self.consumer:
                try:
                    request_data = message.value
                    request_id = request_data.get('request_id', 'unknown')

                    logger.info(f"📨 Received message for request: {request_id}")
                    logger.debug(f"Message details: partition={message.partition}, offset={message.offset}")

                    # Process the request
                    result = self.process_inference_request(request_data)

                    # Send result back
                    self.send_result(result)

                    # Flush producer to ensure delivery
                    self.producer.flush()

                except json.JSONDecodeError as e:
                    logger.error(f"❌ Invalid JSON in message: {e}")
                    ERROR_COUNT.labels(error_type='json_decode').inc()

                except Exception as e:
                    logger.error(f"❌ Error processing message: {e}")
                    logger.exception("Full error traceback:")
                    ERROR_COUNT.labels(error_type='message_processing').inc()

        except KeyboardInterrupt:
            logger.info("🛑 Consumer stopped by user")
        except Exception as e:
            logger.error(f"❌ Consumer error: {e}")
            logger.exception("Full error traceback:")
            raise
        finally:
            logger.info("🔒 Closing consumer and producer...")
            self.consumer.close()
            self.producer.close()
            logger.info("✅ Consumer and producer closed successfully")


def test_consumer():
    """Test the consumer with a sample message"""
    print("🧪 Testing GPT-4.1 nano Triton consumer...")

    try:
        consumer = GPT4NanoTritonConsumer()

        # Test with sample data
        test_requests = [
            {
                'request_id': 'test-001',
                'payload': {
                    'prompt': 'What is artificial intelligence? Explain in 50 words.',
                    'max_tokens': 80,
                    'temperature': 0.7,
                    'top_p': 1.0
                }
            },
            {
                'request_id': 'test-002',
                'payload': {
                    'prompt': 'Write a Python function to calculate factorial.',
                    'max_tokens': 150,
                    'temperature': 0.3,
                    'stop': ['\n\n', 'def ']
                }
            }
        ]

        for test_request in test_requests:
            print(f"\n🧪 Testing request: {test_request['request_id']}")
            result = consumer.process_inference_request(test_request)
            print(f"📋 Test result: {json.dumps(result, indent=2)}")
            print("-" * 50)

    except Exception as e:
        print(f"❌ Test failed: {e}")


def show_metrics():
    """Display current metrics"""
    try:
        metrics_output = prometheus_client.generate_latest()
        print("📊 Current Metrics:")
        print(metrics_output.decode('utf-8'))
    except Exception as e:
        print(f"❌ Failed to get metrics: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            test_consumer()
        elif sys.argv[1] == 'metrics':
            show_metrics()
        else:
            print("Usage: python consumer.py [test|metrics]")
    else:
        consumer = GPT4NanoTritonConsumer()
        consumer.run()

