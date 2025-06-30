# Triton Inference Server Troubleshooting Guide

This guide helps resolve common issues when setting up and running Triton Inference Server with GPT-4.1 nano.

## Common Startup Issues

### 1. "failed to stat file /models"

**Error**: 
```
triton-1 | error: creating server: Internal - failed to stat file /models
```

**Causes & Solutions**:

- **Missing models directory**: Ensure the `models/` directory exists in your project root
  ```bash
  mkdir -p models/gpt4_nano/1
  ```

- **Incorrect Docker volume mount**: Check `docker-compose.yml` has:
  ```yaml
  volumes:
    - ./models:/models
  ```

- **Permission issues**: Fix permissions on the models directory:
  ```bash
  chmod -R 755 models/
  ```

- **Wrong working directory**: Ensure you're running `docker compose up` from the project root where `models/` exists

### 2. "no model configuration provided"

**Error**:
```
error: failed to load model 'gpt4_nano': no model configuration provided
```

**Solutions**:
- Ensure `config.pbtxt` exists in `models/gpt4_nano/`
- Verify the configuration file syntax is valid
- Check file permissions are readable

### 3. "no version subdirectory provided"

**Error**:
```
error: failed to load model 'gpt4_nano': no version subdirectory provided
```

**Solutions**:
- Create version directory: `mkdir -p models/gpt4_nano/1`
- Add `model.py` file in the version directory
- Ensure the version directory is numbered (e.g., `1`, `2`, not `latest`)

## Model Loading Issues

### 4. Python Backend Errors

**Error**:
```
error: failed to load model 'gpt4_nano': python backend requires 'model.py' file
```

**Solutions**:
- Ensure `model.py` exists in `models/gpt4_nano/1/`
- Verify the file contains a `TritonPythonModel` class
- Check Python syntax is valid

### 5. Import Errors in model.py

**Error**:
```
ImportError: No module named 'triton_python_backend_utils'
```

**Solutions**:
- This is normal in local testing; the module is available inside Triton
- For local testing, mock the module or run tests against the Triton container

### 6. Model Configuration Validation

**Error**:
```
error: failed to load model 'gpt4_nano': invalid model configuration
```

**Solutions**:
- Validate `config.pbtxt` syntax
- Ensure input/output specifications match your model implementation
- Check data types are correct (TYPE_STRING, TYPE_INT32, TYPE_FP32)

## Runtime Issues

### 7. Inference Request Failures

**Error**:
```
error: inference request to model 'gpt4_nano' failed: 400 Bad Request
```

**Solutions**:
- Verify input names match config.pbtxt (prompt, max_tokens, temperature)
- Check input data types and shapes
- Ensure required inputs are provided
- Validate JSON request format

### 8. Connection Refused

**Error**:
```
requests.exceptions.ConnectionError: HTTPConnectionPool(host='localhost', port=8000)
```

**Solutions**:
- Check Triton container is running: `docker compose ps`
- Verify port mapping in docker-compose.yml
- Wait for Triton to fully start (can take 30-60 seconds)
- Check Docker network connectivity

### 9. Consumer Service Issues

**Error**:
```
tritonclient._client.InferenceServerException: failed to get model config
```

**Solutions**:
- Ensure model name matches exactly: `gpt4_nano` not `openai_gpt`
- Verify Triton URL in consumer environment: `TRITON_URL=http://triton:8000`
- Check network connectivity between containers

## Debugging Commands

### Check Triton Server Status
```bash
# Health check
curl http://localhost:8000/v2/health/ready

# List models
curl http://localhost:8000/v2/models

# Get model config
curl http://localhost:8000/v2/models/gpt4_nano/config

# Model ready status
curl http://localhost:8000/v2/models/gpt4_nano/ready
```

### Check Docker Services
```bash
# View all services
docker compose ps

# Check Triton logs
docker compose logs triton

# Check consumer logs  
docker compose logs consumer

# Restart specific service
docker compose restart triton
```

### Test Model Directly
```bash
# Simple inference test
curl -X POST http://localhost:8000/v2/models/gpt4_nano/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "prompt",
        "datatype": "BYTES",
        "shape": [1], 
        "data": ["Hello world"]
      }
    ]
  }'
```

### Validate Model Files
```bash
# Check model directory structure
find models/ -type f -ls

# Validate config syntax (basic check)
python -c "
import configparser
with open('models/gpt4_nano/config.pbtxt', 'r') as f:
    content = f.read()
    print('Config file readable:', len(content) > 0)
"

# Check Python model syntax
python -m py_compile models/gpt4_nano/1/model.py
```

## Performance Issues

### 10. Slow Inference Times

**Solutions**:
- Increase instance count in config.pbtxt
- Enable dynamic batching for better throughput
- Consider GPU backend for large models
- Monitor resource usage: `docker stats`

### 11. Memory Issues

**Error**:
```
error: failed to allocate memory for model
```

**Solutions**:
- Increase Docker memory limits
- Reduce batch size in configuration
- Optimize model implementation
- Consider model quantization

## Best Practices for Troubleshooting

1. **Start Simple**: Test with basic inference requests first
2. **Check Logs**: Always examine Triton container logs for detailed errors
3. **Validate Step by Step**: 
   - Model directory structure
   - Configuration file
   - Model implementation
   - Network connectivity
4. **Use Health Endpoints**: Monitor `/v2/health/ready` and `/v2/models/{model}/ready`
5. **Test Incrementally**: Add complexity gradually rather than debugging a full system

## Getting Help

If issues persist:

1. **Check Triton Documentation**: [NVIDIA Triton Docs](https://docs.nvidia.com/deeplearning/triton-inference-server/)
2. **Examine Model Examples**: [Triton Examples](https://github.com/triton-inference-server/server/tree/main/docs/examples)
3. **Review Backend Documentation**: [Python Backend](https://github.com/triton-inference-server/python_backend)

## Environment-Specific Notes

### Development Environment
- Use CPU backend for development and testing
- Mock responses for faster iteration
- Enable verbose logging for debugging

### Production Environment  
- Consider GPU backend for performance
- Implement proper error handling
- Add monitoring and alerting
- Use health checks for container orchestration