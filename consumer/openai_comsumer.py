import tritonclient.http as httpclient
import numpy as np
import json


class OpenAITritonService:
    def __init__(self, triton_url: str = "localhost:8000"):
        self.triton_url = triton_url
        self.client = httpclient.InferenceServerClient(url=triton_url)

    def generate_text(self, prompt: str, max_tokens: int = 150, temperature: float = 0.7):
        """Generate text using OpenAI model through Triton"""

        # Prepare inputs
        inputs = [
            httpclient.InferInput("prompt", [1], "BYTES"),
            httpclient.InferInput("max_tokens", [1], "INT32"),
            httpclient.InferInput("temperature", [1], "FP32")
        ]

        # Set data
        inputs[0].set_data_from_numpy(np.array([prompt.encode('utf-8')], dtype=object))
        inputs[1].set_data_from_numpy(np.array([max_tokens], dtype=np.int32))
        inputs[2].set_data_from_numpy(np.array([temperature], dtype=np.float32))

        # Define outputs
        outputs = [
            httpclient.InferRequestedOutput("response"),
            httpclient.InferRequestedOutput("tokens_used")
        ]

        # Make inference
        response = self.client.infer(
            model_name="openai_gpt",
            inputs=inputs,
            outputs=outputs
        )

        # Extract results
        response_text = response.as_numpy("response")[0].decode('utf-8')
        tokens_used = response.as_numpy("tokens_used")[0]

        return {
            "response": response_text,
            "tokens_used": int(tokens_used)
        }