# Triton Model Repository Setup Guide

This guide explains how to set up and configure the Triton Inference Server model repository for GPT-4.1 nano.

## Overview

The Triton Inference Server requires a specific directory structure for the model repository. This setup provides:

- **Model Name**: `gpt4_nano` 
- **Triton URL**: `http://localhost:8000`
- **API**: Triton v2 inference API
- **Backend**: Python backend for custom model implementation

## Directory Structure

The model repository follows Triton's required structure:

```
models/
└── gpt4_nano/
    ├── config.pbtxt          # Model configuration
    └── 1/                    # Version directory
        └── model.py          # Python model implementation
```

## Model Configuration

The `config.pbtxt` file defines:

### Inputs
- **prompt** (TYPE_STRING): Text prompt for generation
- **max_tokens** (TYPE_INT32, optional): Maximum tokens to generate (default: 150)
- **temperature** (TYPE_FP32, optional): Sampling temperature (default: 0.7)

### Outputs
- **response** (TYPE_STRING): Generated text response
- **tokens_used** (TYPE_INT32): Number of tokens consumed

### Performance Settings
- **Batch Size**: Up to 8 requests
- **Dynamic Batching**: Enabled with preferred sizes [4, 8]
- **Instances**: 2 CPU instances for load balancing

## Docker Compose Configuration

The `docker-compose.yml` includes:

```yaml
triton:
  image: nvcr.io/nvidia/tritonserver:24.01-py3
  ports:
    - "8000:8000"  # HTTP
    - "8001:8001"  # GRPC  
    - "8002:8002"  # Metrics
  volumes:
    - ./models:/models
  command: tritonserver --model-repository=/models --allow-http=true
```

## Starting the Services

1. **Start all services**:
   ```bash
   docker compose up -d
   ```

2. **Check Triton server status**:
   ```bash
   curl http://localhost:8000/v2/health/ready
   ```

3. **List available models**:
   ```bash
   curl http://localhost:8000/v2/models
   ```

## Testing the Model

### Using the provided test script:
```bash
python model.py
```

### Using curl directly:
```bash
curl -X POST http://localhost:8000/v2/models/gpt4_nano/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "prompt",
        "datatype": "BYTES", 
        "shape": [1],
        "data": ["What is machine learning?"]
      }
    ]
  }'
```

### Using the consumer service:
```python
from openai_comsumer import OpenAITritonService

service = OpenAITritonService("http://localhost:8000")
result = service.generate_text("What is artificial intelligence?")
print(result)
```

## Model Implementation Details

The GPT-4.1 nano model (`models/gpt4_nano/1/model.py`) includes:

- **Mock GPT-4.1 nano responses** for testing and development
- **Realistic token counting** based on response length
- **Content-aware responses** for common topics
- **Proper error handling** and logging
- **Triton Python backend integration**

## Integration with Kafka Consumer

The consumer service (`consumer/openai_comsumer.py`) connects to:
- **Triton URL**: `http://triton:8000` (internal Docker network)
- **Model Name**: `gpt4_nano`
- **Expected inputs**: prompt, max_tokens, temperature
- **Expected outputs**: response, tokens_used

## Environment Variables

Key environment variables for the consumer:
- `TRITON_URL`: Triton server endpoint (default: http://triton:8000)
- `KAFKA_BOOTSTRAP`: Kafka bootstrap servers
- `PROMETHEUS_MULTIPROC_DIR`: Metrics directory

## Monitoring and Metrics

Triton provides built-in metrics at:
- **Health**: `http://localhost:8000/v2/health/ready`
- **Model status**: `http://localhost:8000/v2/models/gpt4_nano/ready`
- **Metrics**: `http://localhost:8002/metrics`

## Upgrading to Real GPT-4.1 Nano

To use a real GPT-4.1 nano model:

1. **Replace the mock implementation** in `model.py` with actual model loading
2. **Add model files** (weights, tokenizer) to the version directory
3. **Update dependencies** in the model environment
4. **Configure GPU support** if needed
5. **Adjust resource limits** in the configuration

Example for real model integration:
```python
def initialize(self, args):
    # Load actual GPT-4.1 nano model
    self.model = load_gpt4_nano_model()
    self.tokenizer = load_tokenizer()

def _generate_response(self, prompt, max_tokens, temperature):
    # Use real model for inference
    tokens = self.tokenizer.encode(prompt)
    output = self.model.generate(
        tokens, 
        max_length=max_tokens,
        temperature=temperature
    )
    return self.tokenizer.decode(output)
```