# Project 1 - Kafka + Triton Inference Server

This project provides a complete setup for running GPT-4.1 nano inference through Triton Inference Server with Kafka message processing.

## Quick Start

1. **Start all services**:
   ```bash
   docker compose up -d
   ```

2. **Verify setup**:
   ```bash
   python validate_setup.py
   ```

3. **Test the model**:
   ```bash
   python model.py
   ```

## Services

- **Kafka**: Message broker for inference requests
- **Triton**: Inference server hosting GPT-4.1 nano model
- **Consumer**: Processes Kafka messages and calls Triton
- **API**: HTTP API for submitting inference requests
- **Prometheus/Grafana**: Monitoring and metrics

## Model Configuration

- **Model Name**: `gpt4_nano`
- **Triton URL**: `http://localhost:8000`
- **API Version**: Triton v2 inference API

## Documentation

- **[Setup Guide](TRITON_SETUP.md)**: Complete setup and configuration instructions
- **[Troubleshooting](TRITON_TROUBLESHOOTING.md)**: Common issues and solutions

## Testing

The consumer expects the following model interface:

### Inputs
- `prompt` (string): Text prompt for generation
- `max_tokens` (int, optional): Maximum tokens to generate
- `temperature` (float, optional): Sampling temperature

### Outputs  
- `response` (string): Generated text response
- `tokens_used` (int): Number of tokens consumed

## Architecture

```
[Kafka] → [Consumer] → [Triton] → [GPT-4.1 Nano Model]
   ↑           ↓
[API]    [Prometheus]
```

The consumer service (`consumer/openai_comsumer.py`) connects Kafka messages to Triton inference requests, enabling scalable AI inference processing.