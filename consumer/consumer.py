import logging
import os, json
import sys

from kafka import KafkaConsumer, KafkaProducer
import requests
import prometheus_client
from prometheus_client import Summary

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger("consumer")

TRITON_URL = os.getenv('TRITON_URL')
bootstrap = os.getenv('KAFKA_BOOTSTRAP')
logger.info(f"boostrap server: {bootstrap} ")
# Metrics
INFER_TIME = Summary('consumer_inference_seconds', 'Time spent in Triton inference')

consumer = KafkaConsumer(
    'inference-requests',
    bootstrap_servers=bootstrap,
    value_deserializer=lambda m: json.loads(m.decode()),
    auto_offset_reset='earliest'
)


for msg in consumer:
    req = msg.value
    logger.info(f"consumer received message: {req}")
    # with INFER_TIME.time():
    #     # Call Triton HTTP API
    #     resp = requests.post(f"{TRITON_URL}/v2/models/my_model/infer", json={
    #         "inputs": [{"name":"input__0","data":req['payload']}]
    #     })
    #     out = resp.json()
    # result = {'request_id': req['request_id'], 'output': out}


