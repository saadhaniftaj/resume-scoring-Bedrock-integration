"""
AWS Bedrock DeepSeek Wrapper Module

Provides interface to AWS Bedrock DeepSeek model using boto3.
Implements the same interface as DeepSeekLLM for seamless swapping.
"""

import json
import os
import boto3
from typing import Optional
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BedrockDeepSeekWrapper:
    """
    AWS Bedrock DeepSeek model wrapper.
    
    This class provides the same interface as the local DeepSeekLLM class,
    allowing seamless swapping between local and cloud-based models.
    
    Environment Variables Required:
        AWS_ACCESS_KEY_ID: AWS access key
        AWS_SECRET_ACCESS_KEY: AWS secret key
        AWS_REGION: AWS region (default: us-east-1)
        BEDROCK_MODEL_ID: Bedrock model ID (default: deepseek-r1-distill-llama-70b)
    """
    
    def __init__(self):
        """Initialize Bedrock client and configuration."""
        logger.info("Initializing AWS Bedrock DeepSeek wrapper...")
        
        # Get AWS configuration from environment
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self.model_id = os.getenv("BEDROCK_MODEL_ID", "deepseek-r1-distill-llama-70b")
        
        # Validate credentials are present
        if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
            raise ValueError(
                "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and "
                "AWS_SECRET_ACCESS_KEY environment variables."
            )
        
        # Initialize Bedrock runtime client
        try:
            self.bedrock_runtime = boto3.client(
                service_name='bedrock-runtime',
                region_name=self.region,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
            logger.info(f"Bedrock client initialized | Region: {self.region} | Model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {str(e)}")
            raise
    
    def ask(self, prompt: str, max_tokens: int = 1024) -> str:
        """Send a prompt to Bedrock DeepSeek and extract the clean JSON reply."""
        try:
            request_body = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0
            }
            logger.debug(f"Sending request to Bedrock | Model: {self.model_id} | Max tokens: {max_tokens}")
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            response_body = json.loads(response['body'].read())
            logger.debug(f"Raw Bedrock response keys: {response_body.keys()}")
            generated_text = None
            if 'content' in response_body:
                if isinstance(response_body['content'], list) and response_body['content']:
                    generated_text = response_body['content'][0].get('text', '')
                elif isinstance(response_body['content'], str):
                    generated_text = response_body['content']
            elif 'choices' in response_body and response_body['choices']:
                choice = response_body['choices'][0]
                if 'message' in choice:
                    generated_text = choice['message'].get('content') or choice['message'].get('reasoning_content', '')
                elif 'text' in choice:
                    generated_text = choice.get('text', '')
            elif 'completion' in response_body:
                generated_text = response_body['completion']
            elif 'output' in response_body:
                generated_text = response_body['output']
            if not generated_text:
                logger.error(f"Could not extract text from Bedrock response. Full response: {json.dumps(response_body, indent=2)}")
                raise ValueError("Empty or no text content in Bedrock response")
            logger.debug(f"Received response from Bedrock | Length: {len(generated_text)} chars")
            logger.debug(f"First 200 chars of generated_text: {generated_text[:200]}")
            json_start = generated_text.find('{')
            json_end = generated_text.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                cleaned = generated_text[json_start:json_end + 1]
                logger.debug(f"Extracted JSON block | Start: {json_start}, End: {json_end}, Length: {len(cleaned)}")
                logger.debug(f"First 200 chars of cleaned JSON: {cleaned[:200]}")
                return cleaned.strip()
            logger.warning(f"No valid JSON object detected in model output. First 500 chars: {generated_text[:500]}")
            return generated_text.strip()
        except Exception as e:
            logger.error(f"Bedrock API call failed: {str(e)}", exc_info=True)
            raise Exception(f"Failed to invoke Bedrock model: {str(e)}")
    
    def __repr__(self):
        return f"BedrockDeepSeekWrapper(region={self.region}, model_id={self.model_id})"

