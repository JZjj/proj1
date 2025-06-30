#!/usr/bin/env python3
"""
Triton Model Repository Verification Script

This script validates the Triton model repository structure and configuration
for the GPT-4.1 nano model setup.
"""

import os
import sys
from pathlib import Path

def check_model_structure():
    """Verify the model repository structure."""
    print("🔍 Checking Triton model repository structure...")
    
    issues = []
    models_dir = Path("models")
    
    # Check main models directory
    if not models_dir.exists():
        issues.append("❌ models/ directory does not exist")
        return issues
    
    # Check gpt4_nano model directory
    model_dir = models_dir / "gpt4_nano"
    if not model_dir.exists():
        issues.append("❌ models/gpt4_nano/ directory does not exist")
        return issues
    
    # Check config.pbtxt
    config_file = model_dir / "config.pbtxt"
    if not config_file.exists():
        issues.append("❌ models/gpt4_nano/config.pbtxt does not exist")
    else:
        print("✅ Configuration file found")
        
        # Basic config validation
        try:
            with open(config_file, 'r') as f:
                config_content = f.read()
                if 'name: "gpt4_nano"' not in config_content:
                    issues.append("❌ Model name in config.pbtxt is not 'gpt4_nano'")
                if 'backend: "python"' not in config_content:
                    issues.append("❌ Backend in config.pbtxt is not 'python'")
                else:
                    print("✅ Configuration content validated")
        except Exception as e:
            issues.append(f"❌ Error reading config.pbtxt: {e}")
    
    # Check version directory
    version_dir = model_dir / "1"
    if not version_dir.exists():
        issues.append("❌ models/gpt4_nano/1/ version directory does not exist")
    else:
        print("✅ Version directory found")
    
    # Check model.py
    model_file = version_dir / "model.py"
    if not model_file.exists():
        issues.append("❌ models/gpt4_nano/1/model.py does not exist")
    else:
        print("✅ Model implementation found")
        
        # Basic Python syntax check
        try:
            with open(model_file, 'r') as f:
                model_content = f.read()
                if 'class TritonPythonModel:' not in model_content:
                    issues.append("❌ TritonPythonModel class not found in model.py")
                if 'def initialize(self, args):' not in model_content:
                    issues.append("❌ initialize method not found in model.py")
                if 'def execute(self, requests):' not in model_content:
                    issues.append("❌ execute method not found in model.py")
                else:
                    print("✅ Model implementation structure validated")
        except Exception as e:
            issues.append(f"❌ Error reading model.py: {e}")
    
    return issues

def check_consumer_integration():
    """Check consumer integration files."""
    print("\n🔍 Checking consumer integration...")
    
    issues = []
    
    # Check consumer files
    consumer_dir = Path("consumer")
    if not consumer_dir.exists():
        issues.append("❌ consumer/ directory does not exist")
        return issues
    
    # Check openai_comsumer.py
    consumer_file = consumer_dir / "openai_comsumer.py"
    if not consumer_file.exists():
        issues.append("❌ consumer/openai_comsumer.py does not exist")
    else:
        try:
            with open(consumer_file, 'r') as f:
                consumer_content = f.read()
                if 'model_name="gpt4_nano"' not in consumer_content:
                    issues.append("❌ Consumer still uses old model name (should be gpt4_nano)")
                else:
                    print("✅ Consumer uses correct model name")
        except Exception as e:
            issues.append(f"❌ Error reading consumer file: {e}")
    
    # Check requirements.txt
    req_file = consumer_dir / "requirements.txt"
    if not req_file.exists():
        issues.append("❌ consumer/requirements.txt does not exist")
    else:
        try:
            with open(req_file, 'r') as f:
                req_content = f.read()
                if 'tritonclient' not in req_content:
                    issues.append("❌ tritonclient dependency missing from requirements.txt")
                else:
                    print("✅ Triton client dependency found")
        except Exception as e:
            issues.append(f"❌ Error reading requirements.txt: {e}")
    
    return issues

def check_docker_configuration():
    """Check Docker configuration."""
    print("\n🔍 Checking Docker configuration...")
    
    issues = []
    
    # Check docker-compose.yml
    compose_file = Path("docker-compose.yml")
    if not compose_file.exists():
        issues.append("❌ docker-compose.yml does not exist")
        return issues
    
    try:
        with open(compose_file, 'r') as f:
            compose_content = f.read()
            if './models:/models' not in compose_content:
                issues.append("❌ Models volume mount not found in docker-compose.yml")
            else:
                print("✅ Models volume mount configured")
                
            if 'tritonserver --model-repository=/models' not in compose_content:
                issues.append("❌ Triton model repository path not configured")
            else:
                print("✅ Triton model repository configured")
    except Exception as e:
        issues.append(f"❌ Error reading docker-compose.yml: {e}")
    
    return issues

def check_test_files():
    """Check test files."""
    print("\n🔍 Checking test files...")
    
    issues = []
    
    # Check model.py test file
    test_file = Path("model.py")
    if not test_file.exists():
        issues.append("❌ model.py test file does not exist")
    else:
        try:
            with open(test_file, 'r') as f:
                test_content = f.read()
                if '/v2/models/gpt4_nano/infer' not in test_content:
                    issues.append("❌ Test file still uses old model name")
                else:
                    print("✅ Test file uses correct model name")
        except Exception as e:
            issues.append(f"❌ Error reading test file: {e}")
    
    return issues

def main():
    """Run all validation checks."""
    print("🚀 Triton GPT-4.1 Nano Model Repository Validation")
    print("=" * 55)
    
    os.chdir(Path(__file__).parent)
    
    all_issues = []
    all_issues.extend(check_model_structure())
    all_issues.extend(check_consumer_integration())
    all_issues.extend(check_docker_configuration())
    all_issues.extend(check_test_files())
    
    print("\n" + "=" * 55)
    if all_issues:
        print("❌ VALIDATION FAILED")
        print(f"Found {len(all_issues)} issue(s):")
        for issue in all_issues:
            print(f"  {issue}")
        sys.exit(1)
    else:
        print("✅ VALIDATION PASSED")
        print("All checks completed successfully!")
        print("\nYou can now start the services with:")
        print("  docker compose up -d")
        print("\nTest the model with:")
        print("  python model.py")

if __name__ == "__main__":
    main()