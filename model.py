import json

import requests


def test_openai_triton():
    """Test OpenAI model through Triton"""

    # Triton inference endpoint
    url = "http://localhost:8000/v2/models/openai_gpt/infer"

    # Prepare request
    payload = {
        "inputs": [
            {
                "name": "prompt",
                "datatype": "BYTES",
                "shape": [1],
                "data": ["What is machine learning?"],
            },
            {"name": "max_tokens", "datatype": "INT32", "shape": [1], "data": [100]},
            {"name": "temperature", "datatype": "FP32", "shape": [1], "data": [0.7]},
        ]
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        print("OpenAI model response:")
        print(json.dumps(result, indent=2))

        # Extract the actual response
        if "outputs" in result:
            for output in result["outputs"]:
                if output["name"] == "response":
                    response_text = output["data"][0]
                    print(f"\nGenerated text: {response_text}")
                elif output["name"] == "tokens_used":
                    tokens = output["data"][0]
                    print(f"Tokens used: {tokens}")

    except requests.exceptions.RequestException as e:
        print(f" Request failed: {e}")
    except Exception as e:
        print(f" Error: {e}")


if __name__ == "__main__":
    test_openai_triton()
