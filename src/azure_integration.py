import os
import logging
import uuid
import json
import re
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import ClientSecretCredential

logger = logging.getLogger(__name__)

class AzureOpenAIClient:
    """Client for Azure OpenAI API integration."""
    
    def __init__(self):
        """Initialize Azure OpenAI client."""
        # Load environment variables
        load_dotenv()
        
        # Azure OpenAI settings
        self.scope = os.getenv("SCOPE")
        self.tenant_id = os.getenv("TENANT_ID")
        self.client_id = os.getenv("CLIENT_ID")
        self.client_secret = os.getenv("CLIENT_SECRET")
        self.subscription_key = os.getenv("SUBSCRIPTION_KEY")
        self.api_version = os.getenv("AZURE_API_VERSION")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        
        # Default settings
        self.default_max_attempts = 3
        self.default_wait_time = 2
        
        # Check if Azure OpenAI is configured
        self.is_configured = all([
            self.tenant_id, 
            self.client_id, 
            self.client_secret, 
            self.subscription_key,
            self.api_version,
            self.azure_endpoint
        ])
        
        if self.is_configured:
            logger.info("Azure OpenAI client initialized successfully")
        else:
            logger.warning("Azure OpenAI client not fully configured. Some environment variables are missing.")
    
    def refresh_token(self) -> str:
        """Refreshes the Azure API token.

        Returns:
            str: The refreshed Azure token
        """
        try:
            # Get token with Azure credentials
            client_credentials = ClientSecretCredential(
                self.tenant_id, 
                self.client_id, 
                self.client_secret
            )
            access_token = client_credentials.get_token(self.scope).token
            logger.info("Successfully refreshed Azure token")
            return access_token

        except Exception as e:
            logger.error("Failed to refresh token: %s", e)
            raise
    
    def get_embeddings(self, text: Union[str, List[str]], model: str) -> List[float]:
        """Get embeddings using Azure OpenAI API.

        Parameters:
            text (Union[str, List[str]]): Text to embed
            model (str): Azure OpenAI model to use

        Returns:
            List[float]: Embedding vector
        """
        if not self.is_configured:
            raise ValueError("Azure OpenAI client not configured. Check environment variables.")
        
        # Convert single text to list
        if isinstance(text, str):
            text_input = [text]
        else:
            text_input = text
        
        for attempt in range(self.default_max_attempts):
            try:
                # Refresh token and create client
                access_token = self.refresh_token()
                if not access_token:
                    raise ValueError("Failed to refresh Azure AD token")
                    
                client = AzureOpenAI(
                    api_version=self.api_version,
                    azure_endpoint=self.azure_endpoint,
                    azure_ad_token=access_token
                )
                
                # Get embeddings from Azure OpenAI
                response = client.embeddings.create(
                    model=model,
                    input=text_input,
                    extra_header={'x-correlation-id': str(uuid.uuid4()), 'x-subscription-key': self.subscription_key}
                )
                
                # Extract embeddings
                if len(text_input) == 1:
                    return response.data[0].embedding
                else:
                    return [item.embedding for item in response.data]
                
            except Exception as e:
                if attempt == self.default_max_attempts - 1:
                    logger.error(f"Failed to get embeddings after {self.default_max_attempts} attempts: {e}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
    
    def get_completion(self,
                      prompt: str,
                      system_prompt: Optional[str] = None,
                      model: Optional[str] = "gpt-4o",
                      temperature: Optional[float] = 0.2,
                      max_tokens: Optional[int] = 1000,
                      top_p: Optional[float] = 0.9,
                      top_k: Optional[int] = 50,
                      frequency_penalty: Optional[float] = 1,
                      presence_penalty: Optional[float] = 1,
                      seed: Optional[int] = None,
                      json_schema: Optional[Dict[str, Any]] = None) -> str:
        """Get text completion using Azure OpenAI API.
        
        Parameters:
            prompt (str): Input prompt
            system_prompt (Optional[str]): System prompt to guide the model
            model (Optional[str]): Azure OpenAI model to use
            temperature (Optional[float]): Temperature for generation
            max_tokens (Optional[int]): Maximum tokens to generate
            top_p (Optional[float]): Top-p sampling parameter
            top_k (Optional[int]): Top-k sampling parameter
            frequency_penalty (Optional[float]): Frequency penalty parameter
            presence_penalty (Optional[float]): Presence penalty parameter
            seed (Optional[int]): Random seed for reproducibility
            json_schema (Optional[Dict[str, Any]]): JSON schema for response validation
            
        Returns:
            str: Generated response
        """
        if not self.is_configured:
            raise ValueError("Azure OpenAI client not configured. Check environment variables.")
        
        # Process prompt into messages
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}]
        else:
            messages = [{"role": "user", "content": prompt}]
        
        # Determine response format
        if json_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": json_schema
            }
        elif re.search(r'\bJSON\b', prompt, re.IGNORECASE):
            response_format = {"type": "json_object"}
        else:
            response_format = {"type": "text"}
            
        logger.info(f"Response format: {response_format}")

        # Allow a number of attempts when calling API
        for attempt in range(self.default_max_attempts):
            try:
                # Refresh token
                access_token = self.refresh_token()
                client = AzureOpenAI(
                    api_version=self.api_version, 
                    azure_endpoint=self.azure_endpoint,
                    azure_ad_token=access_token
                )
                
                # Make API call
                response = client.chat.completions.create(
                    engine=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    top_k=top_k,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    response_format=response_format,
                    seed=seed,
                    extra_header={
                        'x-correlation-id': str(uuid.uuid4()),
                        'x-subscription-key': self.subscription_key
                    }
                )
                
                # Extract response text
                response_text = response.choices[0].message.content
                
                # Process JSON response if needed
                if response_format["type"] == "json_object":
                    try:
                        response_text = json.loads(response_text.strip('```json').strip('```'))
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON response: {e}")
                        raise ValueError("Invalid JSON response from model")
                
                elif response_format["type"] == "text":
                    try:
                        response_text = response.choices[0].message.content
                    except Exception as e:
                        logger.error(f"Failed to extract text response: {e}")
                        raise ValueError("Invalid text response from model")
                return response_text
                
            except Exception as e:
                logger.warning("Azure attempt %d failed: %s", attempt + 1, e)
                if attempt < self.default_max_attempts - 1:
                    self.refresh_token()
                else:
                    raise
