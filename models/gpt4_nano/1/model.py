import json
import numpy as np
import triton_python_backend_utils as pb_utils
from typing import List
import time
import logging

class TritonPythonModel:
    """
    GPT-4.1 Nano model implementation for Triton Inference Server.
    This is a mock implementation that simulates GPT-4.1 nano responses.
    """

    def initialize(self, args):
        """Initialize the model."""
        self.model_config = model_config = json.loads(args['model_config'])
        
        # Get output configuration
        output_configs = pb_utils.get_output_config_by_name(
            model_config, "response")
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_configs['data_type'])
        
        output_configs_tokens = pb_utils.get_output_config_by_name(
            model_config, "tokens_used")
        self.output_dtype_tokens = pb_utils.triton_string_to_numpy(
            output_configs_tokens['data_type'])
        
        # Initialize logger
        self.logger = pb_utils.Logger
        self.logger.log_info("GPT-4.1 Nano model initialized successfully")

    def execute(self, requests):
        """Execute inference on a batch of requests."""
        responses = []
        
        for request in requests:
            # Extract inputs
            prompt = pb_utils.get_input_tensor_by_name(request, "prompt")
            max_tokens_tensor = pb_utils.get_input_tensor_by_name(request, "max_tokens")
            temperature_tensor = pb_utils.get_input_tensor_by_name(request, "temperature")
            
            # Convert inputs to numpy
            prompt_text = prompt.as_numpy()[0].decode('utf-8') if prompt else ""
            max_tokens = max_tokens_tensor.as_numpy()[0] if max_tokens_tensor else 150
            temperature = temperature_tensor.as_numpy()[0] if temperature_tensor else 0.7
            
            # Simulate GPT-4.1 nano inference
            response_text, tokens_used = self._generate_response(
                prompt_text, max_tokens, temperature
            )
            
            # Create output tensors
            response_tensor = pb_utils.Tensor(
                "response",
                np.array([response_text.encode('utf-8')], dtype=object)
            )
            
            tokens_tensor = pb_utils.Tensor(
                "tokens_used",
                np.array([tokens_used], dtype=np.int32)
            )
            
            # Create response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[response_tensor, tokens_tensor]
            )
            responses.append(inference_response)
        
        return responses

    def _generate_response(self, prompt: str, max_tokens: int, temperature: float) -> tuple:
        """
        Mock GPT-4.1 nano text generation.
        In a real implementation, this would call the actual GPT-4.1 nano model.
        """
        # Simulate processing time
        time.sleep(0.1)
        
        # Mock responses based on prompt content for realistic testing
        if "machine learning" in prompt.lower():
            response = (
                "Machine learning is a subset of artificial intelligence (AI) that enables "
                "computers to learn and improve from experience without being explicitly programmed. "
                "It involves algorithms that can identify patterns in data and make predictions "
                "or decisions based on that analysis."
            )
        elif "python" in prompt.lower():
            response = (
                "Python is a high-level, interpreted programming language known for its "
                "simplicity and readability. It's widely used in data science, web development, "
                "automation, and artificial intelligence applications."
            )
        elif "triton" in prompt.lower():
            response = (
                "Triton Inference Server is an open-source inference serving software that "
                "enables teams to deploy AI models from multiple frameworks at scale. "
                "It provides optimized performance for both CPU and GPU inference."
            )
        else:
            # Generic response for other prompts
            response = (
                f"I understand you're asking about: '{prompt[:50]}...' "
                "This is a response from GPT-4.1 nano running on Triton Inference Server. "
                "The model is designed to provide helpful and informative responses to your queries."
            )
        
        # Simulate token usage (roughly words * 1.3 for realistic token count)
        estimated_tokens = len(response.split()) * 1.3
        tokens_used = min(int(estimated_tokens), max_tokens)
        
        # Truncate response if it exceeds max_tokens
        if tokens_used >= max_tokens:
            words = response.split()
            truncated_words = words[:int(max_tokens / 1.3)]
            response = " ".join(truncated_words) + "..."
            tokens_used = max_tokens
        
        return response, int(tokens_used)

    def finalize(self):
        """Clean up resources."""
        self.logger.log_info("GPT-4.1 Nano model finalized")