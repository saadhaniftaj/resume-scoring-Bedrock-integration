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
        """
        Send a prompt to the Bedrock DeepSeek model.
        
        This method implements the same interface as DeepSeekLLM.ask(),
        allowing it to be a drop-in replacement.
        
        Args:
            prompt: The prompt text (can include system + user prompts combined)
            max_tokens: Maximum tokens to generate (default: 1024)
        
        Returns:
            The model's response text
        
        Raises:
            Exception: If the Bedrock API call fails
        """
        try:
            # Construct request body for Bedrock DeepSeek
            # DeepSeek models on Bedrock typically expect a messages format
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
            
            # Invoke the model
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            
            # Parse the response
            response_body = json.loads(response['body'].read())
            logger.debug(f"Raw Bedrock response keys: {response_body.keys()}")
            
            # Extract the generated text
            # DeepSeek R1 on Bedrock can have multiple response formats
            generated_text = None
            
            if 'content' in response_body:
                if isinstance(response_body['content'], list) and len(response_body['content']) > 0:
                    # Format: {"content": [{"text": "..."}]}
                    generated_text = response_body['content'][0].get('text', '')
                elif isinstance(response_body['content'], str):
                    # Format: {"content": "..."}
                    generated_text = response_body['content']
            elif 'choices' in response_body and len(response_body['choices']) > 0:
                # Format: {"choices": [{"message": {"content": "...", "reasoning_content": "..."}}]}
                choice = response_body['choices'][0]
                if 'message' in choice:
                    # DeepSeek R1 returns reasoning_content (thinking) and content (final answer)
                    # Prefer content, but fall back to reasoning_content if content is null
                    generated_text = choice['message'].get('content') or choice['message'].get('reasoning_content', '')
                elif 'text' in choice:
                    generated_text = choice.get('text', '')
            elif 'completion' in response_body:
                # Format: {"completion": "..."}
                generated_text = response_body['completion']
            elif 'output' in response_body:
                # Format: {"output": "..."}
                generated_text = response_body['output']
            
            if generated_text is None or generated_text == '':
                logger.error(f"Could not extract text from Bedrock response. Full response: {json.dumps(response_body, indent=2)}")
                raise ValueError("Empty or no text content in Bedrock response")
            
            logger.debug(f"Received response from Bedrock | Length: {len(generated_text)} chars")
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Bedrock API call failed: {str(e)}", exc_info=True)
            raise Exception(f"Failed to invoke Bedrock model: {str(e)}")
    
    def __repr__(self):
        return f"BedrockDeepSeekWrapper(region={self.region}, model_id={self.model_id})"

