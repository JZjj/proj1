import json
import logging
import os
import sys
import threading

import prometheus_client
from fastapi import FastAPI, HTTPException
from kafka import KafkaProducer, KafkaConsumer
from prometheus_client import Summary, Counter
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger("api")


# ——————  Models ——————
class InferenceRequest(BaseModel):
    request_id: str
    payload: dict


class InferenceResult(BaseModel):
    request_id: str
    output: dict


# —————— Metrics ——————
REQUEST_TIME = Summary("api_request_processing_seconds", "API request latency")
REQUEST_COUNT = Counter("api_request_total", "Total API requests")

# —————— App & Kafka setup ——————
app = FastAPI()
bootstrap = os.getenv("KAFKA_BOOTSTRAP")
logging.info(f"Bootstrap server : {bootstrap}")
producer = KafkaProducer(
    bootstrap_servers=bootstrap, value_serializer=lambda v: json.dumps(v).encode()
)

# Background thread to consume results
results = {}


def consume_results():
    consumer = KafkaConsumer(
        "inference-results",
        bootstrap_servers=bootstrap,
        value_deserializer=lambda m: json.loads(m.decode()),
        auto_offset_reset="earliest",
    )
    for msg in consumer:
        res = msg.value
        results[res["request_id"]] = res


threading.Thread(target=consume_results, daemon=True).start()


# —————— Endpoints ——————
@app.post("/infer", response_model=InferenceRequest)
@REQUEST_COUNT.count_exceptions()
@REQUEST_TIME.time()
def enqueue(req: InferenceRequest):
    """Accepts request and publishes to Kafka."""
    logger.info(f"Inference request: {req}")
    producer.send("inference-requests", value=req.dict())
    return req


@app.get("/result/{request_id}", response_model=InferenceResult)
def get_result(request_id: str):
    """Polls in-memory dict for result."""
    if request_id not in results:
        raise HTTPException(404, f"No result for {request_id}")
    return InferenceResult(**results.pop(request_id))


@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return prometheus_client.generate_latest()
