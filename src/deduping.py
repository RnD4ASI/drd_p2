import os
import json
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
import uuid
import re
import datetime
import time
import traceback
import functools
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from src.azure_integration import AzureOpenAIClient

logger = logging.getLogger(__name__)

def log_execution_time(func):
    """Decorator to log the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting {func.__name__}")
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed {func.__name__} in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {execution_time:.2f} seconds: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise
    return wrapper

class DuplicateDetector:
    """Detects duplicate data attributes using language models."""
    
    @log_execution_time
    def __init__(self, model_path: str = "model/qwen3-1.7B", provider: str = "huggingface", tmp_responses_dir: Optional[Union[str, Path]] = None):
        """Initialize the DuplicateDetector.
        
        Parameters:
            model_path (str): Path to the language model or name of Azure OpenAI model
            provider (str): Model provider, either 'huggingface' or 'azure_openai'
            tmp_responses_dir (Optional[Union[str, Path]]): Directory to save raw LLM responses. Defaults to None.
        """
        logger.info(f"Initializing DuplicateDetector with model_path={model_path}, provider={provider}")
        
        self.model_path = model_path
        self.provider = provider.lower()
        self.model = None
        self.tokenizer = None
        self.azure_client = None
        self.tmp_responses_dir = None
        self.memory_usage_start = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        logger.info(f"Initial memory usage: {self.memory_usage_start:.2f} MB")
        
        if tmp_responses_dir:
            self.tmp_responses_dir = Path(tmp_responses_dir)
            logger.info(f"Setting up tmp_responses_dir at {self.tmp_responses_dir}")
            os.makedirs(self.tmp_responses_dir, exist_ok=True)
        else:
            logger.debug("No tmp_responses_dir provided, raw LLM responses will not be saved")
        
        # Initialize Azure client if using Azure OpenAI
        if self.provider == "azure_openai":
            logger.info("Initializing Azure OpenAI client")
            self.azure_client = AzureOpenAIClient()
            if not self.azure_client.is_configured:
                logger.warning("Azure OpenAI not fully configured. Will fall back to HuggingFace if Azure completion is requested.")
            else:
                logger.info("Azure OpenAI client successfully configured")
        
        logger.info("DuplicateDetector initialization complete")
    
    @log_execution_time
    def load_model(self):
        """Load the language model."""
        try:
            if not self.model_path:
                logger.error("Model path not provided")
                raise ValueError("Model path not provided")
            
            # Log system resources before loading
            mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
            logger.info(f"Memory usage before model loading: {mem_before:.2f} MB")
            
            # If using Azure OpenAI, no need to load a local model
            if self.provider == "azure_openai":
                logger.info(f"Using Azure OpenAI language model: {self.model_path}")
                return
            
            logger.info(f"Loading language model from {self.model_path}")
            
            # Check if this is a local path
            local_path = os.path.join(os.getcwd(), self.model_path)
            if os.path.exists(local_path):
                logger.info(f"Found local model at {local_path}")
                # Use absolute path for local models
                model_path_to_use = os.path.abspath(local_path)
                logger.info(f"Using absolute path: {model_path_to_use}")
                
                # Check if the directory has required model files for a HuggingFace model
                required_files = ["config.json", "model.safetensors", "tokenizer.json"]
                existing_files = [f for f in required_files if os.path.exists(os.path.join(model_path_to_use, f))]
                files_exist = len(existing_files) == len(required_files)
                
                logger.info(f"Checking for required model files in {model_path_to_use}:")
                for file in required_files:
                    file_path = os.path.join(model_path_to_use, file)
                    logger.info(f"  - {file}: {'EXISTS' if os.path.exists(file_path) else 'MISSING'}")
                
                if files_exist:
                    logger.info("✅ All required model files found. Proceeding with local loading.")
                    # First approach: Direct loading with local_files_only flag
                    try:
                        logger.info("🔄 METHOD 1: Attempting to load model using direct path with local_files_only=True")
                        start_time = time.time()
                        self.tokenizer = AutoTokenizer.from_pretrained(model_path_to_use, local_files_only=True)
                        logger.info("  - Tokenizer loaded successfully")
                        
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_path_to_use,
                            device_map="auto",
                            trust_remote_code=True,
                            local_files_only=True
                        )
                        load_time = time.time() - start_time
                        logger.info(f"✅ METHOD 1: Successfully loaded model using direct path with local_files_only=True (took {load_time:.2f}s)")
                        logger.info(f"  - Model loaded to device: {self.model.device}")
                        return
                    except Exception as e:
                        logger.warning(f"❌ METHOD 1: Failed to load model directly: {str(e)}")
                        
                    # Second approach: Try loading with additional parameters
                    try:
                        logger.info("🔄 METHOD 2: Attempting to load model with additional parameters...")
                        start_time = time.time()
                        
                        logger.info("  - Loading tokenizer with use_fast=False")
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_path_to_use, 
                            local_files_only=True,
                            use_fast=False
                        )
                        logger.info("  - Tokenizer loaded successfully")
                        
                        logger.info("  - Loading model with low_cpu_mem_usage=True")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_path_to_use,
                            device_map="auto",
                            trust_remote_code=True,
                            local_files_only=True,
                            low_cpu_mem_usage=True
                        )
                        load_time = time.time() - start_time
                        logger.info(f"✅ METHOD 2: Successfully loaded model with additional parameters (took {load_time:.2f}s)")
                        logger.info(f"  - Model loaded to device: {self.model.device}")
                        logger.info(f"  - Model type: {type(self.model).__name__}")
                        return
                    except Exception as e:
                        logger.error(f"❌ METHOD 2: Failed to load model with additional parameters: {str(e)}")
                        raise
                else:
                    logger.warning(f"❌ Required model files not found in {model_path_to_use}. Found {len(existing_files)}/{len(required_files)} required files.")
            
            # If not a local path or local files don't exist, try to download from HuggingFace
            logger.info(f"🔄 METHOD 3: Attempting to load model from HuggingFace: {self.model_path}")
            start_time = time.time()
            
            logger.info("  - Downloading and loading tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            logger.info("  - Downloading and loading model")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                trust_remote_code=True
            )
            load_time = time.time() - start_time
            logger.info(f"✅ METHOD 3: Successfully loaded model from HuggingFace (took {load_time:.2f}s)")
            logger.info(f"  - Model type: {type(self.model).__name__}")
            logger.info(f"  - Model loaded to device: {self.model.device}")
            
            # Optionally save the model locally for future use
            if local_path and not os.path.exists(local_path):
                logger.info(f"💾 Saving model to {local_path} for future use")
                start_time = time.time()
                os.makedirs(local_path, exist_ok=True)
                
                logger.info("  - Saving model weights and configuration")
                self.model.save_pretrained(local_path)
                
                logger.info("  - Saving tokenizer files")
                self.tokenizer.save_pretrained(local_path)
                
                save_time = time.time() - start_time
                logger.info(f"✅ Model and tokenizer saved to {local_path} (took {save_time:.2f}s)")
                
                # Log saved files with sizes
                logger.info(f"📁 Model files saved to disk:")
                total_size_mb = 0
                for root, dirs, files in os.walk(local_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                        total_size_mb += file_size
                        logger.info(f"  - {os.path.relpath(file_path, local_path)}: {file_size:.2f} MB")
                logger.info(f"  Total model size on disk: {total_size_mb:.2f} MB")
            logger.info("Language model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading language model: {e}")
            raise
    
    @log_execution_time
    def generate_response(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.2, repeat_penalty: float = 1.2, provider: str = None) -> str:
        """Generate a response from a selected language model.
        
        Parameters:
            prompt (str): Input prompt
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            repeat_penalty (float): Penalty for token repetition
            provider (str, optional): Model provider to use ('huggingface' or 'azure_openai'). If None, uses the instance's provider.
            
        Returns:
            str: Generated response
        """
        # Use the provided provider if specified, otherwise use the instance's provider
        current_provider = provider if provider else self.provider
        
        # Log prompt details (truncated for brevity)
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        logger.info(f"Generating response with {current_provider} model {self.model_path}")
        logger.info(f"Parameters: max_tokens={max_tokens}, temp={temperature}, repeat_penalty={repeat_penalty}")
        logger.debug(f"Prompt preview: {prompt_preview}")
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        # Track memory usage
        mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        logger.debug(f"Memory usage before generation: {mem_before:.2f} MB")
        
        generation_start = time.time()
        try:
            # Use Azure OpenAI for completions if specified
            if current_provider == "azure_openai":
                if not self.azure_client or not self.azure_client.is_configured:
                    logger.error("Azure OpenAI client not configured. Check environment variables.")
                    raise ValueError("Azure OpenAI client not configured. Check environment variables.")
                
                # Get completion from Azure OpenAI
                try:
                    logger.info("Sending request to Azure OpenAI API")
                    system_prompt = "You are an helful AI assistant. You analyse the tasks given, think step by step and respond in a valid JSON format."
                    response = self.azure_client.get_completion(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=self.model_path,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    # Log response details
                    generation_time = time.time() - generation_start
                    response_preview = response[:100] + "..." if len(response) > 100 else response
                    logger.info(f"Azure OpenAI generation completed in {generation_time:.2f}s")
                    logger.info(f"Response length: {len(response)} characters")
                    logger.debug(f"Response preview: {response_preview}")
                    
                    return response
                except Exception as e:
                    logger.error(f"Azure OpenAI completion failed: {e}")
                    logger.error(f"Stack trace: {traceback.format_exc()}")
                    raise
            
            # Use local HuggingFace model
            else:
                if self.model is None or self.tokenizer is None:
                    logger.error("Model not loaded. Call load_model first.")
                    raise ValueError("Model not loaded. Call load_model first.")
                
                logger.info("Tokenizing prompt for local model generation")
                # Tokenize the prompt
                tokenize_start = time.time()
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                tokenize_time = time.time() - tokenize_start
                
                # Log tokenization details
                prompt_token_count = inputs.input_ids.shape[1]
                logger.info(f"Prompt tokenized in {tokenize_time:.2f}s, contains {prompt_token_count} tokens")
                
                # Generate response
                logger.info(f"Starting model generation on device {self.model.device}...")
                generation_config = GenerationConfig(
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    repeat_penalty=repeat_penalty,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Track memory before generation
                mem_before_gen = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
                
                # Start generation timer
                gen_start_time = time.time()
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
                gen_time = time.time() - gen_start_time
                
                # Track memory after generation
                mem_after_gen = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
                mem_diff = mem_after_gen - mem_before_gen
                
                # Extract the generated part (remove the prompt)
                prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
                response_tokens = self.tokenizer.decode(outputs[0][prompt_tokens:], skip_special_tokens=True)
                
                # Log generation statistics
                output_token_count = outputs.shape[1] - prompt_token_count
                tokens_per_second = output_token_count / gen_time if gen_time > 0 else 0
                
                logger.info(f"Model generation complete in {gen_time:.2f}s")
                logger.info(f"Generated {output_token_count} new tokens at {tokens_per_second:.2f} tokens/second")
                logger.info(f"Response length: {len(response_tokens)} characters")
                logger.info(f"Memory usage: before={mem_before_gen:.2f}MB, after={mem_after_gen:.2f}MB, diff={mem_diff:.2f}MB")
                
                response_preview = response_tokens[:100] + "..." if len(response_tokens) > 100 else response_tokens
                logger.debug(f"Response preview: {response_preview}")
                
                return response_tokens.strip()
            
        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            raise
    
    @log_execution_time
    def save_response(self, context_name: str, response_text: str):
        """Saves the raw LLM response to a file if tmp_responses_dir is set."""
        if not self.tmp_responses_dir:
            logger.warning("tmp_responses_dir is not set. Cannot save LLM response.")
            return
        
        try:
            # Log response details
            response_length = len(response_text)
            logger.info(f"Saving LLM response for context '{context_name}' (length: {response_length} chars)")
            
            # Sanitize context_name for filename
            safe_context_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', context_name)
            max_len_context = 100 # Max length for context part of filename
            if len(safe_context_name) > max_len_context:
                logger.debug(f"Context name too long ({len(safe_context_name)} chars), truncating to {max_len_context} chars")
                safe_context_name = safe_context_name[:max_len_context]

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
            unique_id = uuid.uuid4().hex[:8]
            filename = self.tmp_responses_dir / f"{safe_context_name}_{timestamp}_{unique_id}.txt"
            
            logger.debug(f"Writing response to file: {filename}")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(response_text)
            
            # Get file size for logging
            file_size = os.path.getsize(filename) / 1024  # KB
            logger.info(f"Saved LLM response to {filename} (size: {file_size:.2f} KB)")
        except Exception as e:
            logger.error(f"Failed to save LLM response for context '{context_name}': {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")

    @log_execution_time
    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response with robust fallback mechanisms for Huggingface/Open Source models.
        Returns a dict with the parsed data or a default structure if extraction fails.
        """
        import re
        import json
        
        # Log response details (truncated for brevity)
        response_preview = response[:100] + "..." if len(response) > 100 else response
        logger.info(f"Extracting JSON from response (length: {len(response)} chars)")
        logger.debug(f"Response preview: {response_preview}")
        
        try:
            # Step 1: Try to extract JSON from code block delimiters
            logger.info("Step 1: Attempting to extract JSON from code block delimiters")
            def extract_json_codeblock(text: str) -> str:
                if "```json" in text:
                    logger.debug("Found ```json marker in response")
                    parts = text.split("```json")
                    if len(parts) > 1 and "```" in parts[1]:
                        extracted = parts[1].split("```")[0].strip()
                        logger.debug(f"Extracted JSON from ```json block (length: {len(extracted)} chars)")
                        return extracted
                if "```" in text:
                    logger.debug("Found ``` marker in response")
                    parts = text.split("```")
                    if len(parts) > 1:
                        extracted = parts[1].strip()
                        logger.debug(f"Extracted content from ``` block (length: {len(extracted)} chars)")
                        return extracted
                logger.debug("No code block markers found, using raw text")
                return text

            json_str = extract_json_codeblock(response)
            try:
                result = json.loads(json_str)
                logger.info("✅ Successfully parsed JSON from code block")
                logger.debug(f"Parsed JSON structure: {list(result.keys()) if isinstance(result, dict) else 'non-dict result'}")
                return result
            except json.JSONDecodeError as e:
                logger.info(f"❌ Code block extraction failed: {str(e)}, trying last JSON object extraction")

            # Step 2: Extract the last, complete, and valid JSON object using regex
            logger.info("Step 2: Attempting to extract last complete JSON object using regex")
            def extract_last_json(text: str) -> str:
                # Find all JSON-like objects in the text
                try:
                    matches = list(re.finditer(r'\{(?:[^{}]|(?R))*\}', text, re.DOTALL))
                    logger.debug(f"Found {len(matches)} potential JSON objects in text")
                    
                    for i, match in enumerate(reversed(matches)):
                        json_candidate = match.group(0)
                        logger.debug(f"Checking JSON candidate {i+1}/{len(matches)} (length: {len(json_candidate)} chars)")
                        try:
                            json.loads(json_candidate)
                            logger.debug(f"Found valid JSON object at position {match.start()}")
                            return json_candidate
                        except json.JSONDecodeError as e:
                            logger.debug(f"Invalid JSON in candidate {i+1}: {str(e)}")
                            continue
                    logger.debug("No valid JSON objects found using regex")
                except Exception as regex_err:
                    logger.warning(f"Error in regex pattern matching: {str(regex_err)}")
                return None

            json_str = extract_last_json(response)
            if json_str:
                try:
                    result = json.loads(json_str)
                    logger.info("✅ Successfully parsed last valid JSON object in response")
                    logger.debug(f"Parsed JSON structure: {list(result.keys()) if isinstance(result, dict) else 'non-dict result'}")
                    return result
                except json.JSONDecodeError as e:
                    logger.info(f"❌ Last JSON object extraction failed: {str(e)}, trying duplicates pattern")

            # Step 3: Try extracting JSON with "duplicates" key (specific to your use case)
            logger.info("Step 3: Attempting to extract JSON with 'duplicates' key pattern")
            pattern = r'\{\s*"duplicates"\s*:\s*\[.*?\]\s*\}'
            matches = re.findall(pattern, response, re.DOTALL)
            logger.debug(f"Found {len(matches)} potential matches for 'duplicates' pattern")
            
            for i, match in enumerate(reversed(matches)):
                try:
                    result = json.loads(match)
                    logger.info(f"✅ Successfully parsed JSON with 'duplicates' key (match {i+1}/{len(matches)})")
                    return result
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse match {i+1}: {str(e)}")
                    continue

            # Step 4: Fallback - attempt to extract structured information from text
            logger.warning("⚠️ No valid JSON found, attempting to extract structured information from text")
            groups = {}
            
            # Try to find group patterns
            group_pattern = r'group(\d+)[^\w].*?:([^\n]+)'
            group_matches = re.findall(group_pattern, response, re.IGNORECASE)
            logger.debug(f"Found {len(group_matches)} group pattern matches")
            
            # Try to find list patterns
            list_pattern = r'- ([\w_]+),\s*([\w_]+)(?:,\s*([\w_]+))?\s*\(group(\d+)\)'
            list_matches = re.findall(list_pattern, response, re.IGNORECASE)
            logger.debug(f"Found {len(list_matches)} list pattern matches")

            # Process group matches
            for group_id, attrs_text in group_matches:
                attrs = re.findall(r'([\w_]+)', attrs_text)
                if attrs:
                    group_name = f"group{group_id}"
                    groups.setdefault(group_name, []).extend(attrs)
                    logger.debug(f"Added {len(attrs)} attributes to {group_name} from group pattern")
            
            # Process list matches
            for match in list_matches:
                attrs = [attr for attr in match[:-1] if attr]
                group_id = match[-1]
                group_name = f"group{group_id}"
                groups.setdefault(group_name, []).extend(attrs)
                logger.debug(f"Added {len(attrs)} attributes to {group_name} from list pattern")

            # If no patterns matched, try section-based extraction
            if not groups:
                logger.debug("No pattern matches found, trying section-based extraction")
                sections = re.split(r'\n\s*\n', response)
                logger.debug(f"Split response into {len(sections)} sections")
                
                for i, section in enumerate(sections):
                    if "duplicate" in section.lower() and "group" in section.lower():
                        logger.debug(f"Found potential duplicate group in section {i+1}")
                        attr_pattern = r'([a-zA-Z_]+(?:_[a-zA-Z_]+)*)'
                        attrs = re.findall(attr_pattern, section)
                        if len(attrs) >= 2:
                            group_name = f"group{len(groups) + 1}"
                            groups[group_name] = attrs
                            logger.debug(f"Created {group_name} with {len(attrs)} attributes from section {i+1}")

            # Build result structure
            result = {"duplicates": []}
            for group_name, attrs in groups.items():
                for attr in attrs:
                    result["duplicates"].append({
                        "name": attr,
                        "is_duplicate": True,
                        "group": group_name
                    })
            
            if result["duplicates"]:
                logger.info(f"✅ Extracted {len(groups)} duplicate groups with {len(result['duplicates'])} total attributes from text fallback")
                return result

            logger.warning("❌ Could not extract structured information, returning default empty structure")
            return {"duplicates": []}
        except Exception as e:
            logger.error(f"❌ Error in extract_json_from_response: {e}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return {"duplicates": []}

    #functional process 1
    @log_execution_time
    def detection(self, cluster_attributes: pd.DataFrame, attributes_df: pd.DataFrame = None, incremental: bool = False, provider: str = None) -> Dict[str, Any]:
        """Detect duplicates within a cluster.
        
        Parameters:
            cluster_attributes (pd.DataFrame): DataFrame containing attributes in a cluster
            attributes_df (pd.DataFrame, optional): DataFrame containing all attributes
            incremental (bool, optional): Flag for incremental processing
            provider (str, optional): Model provider ('huggingface' or 'azure_openai'). If None, uses the instance's provider.
            
        Returns:
            Dict[str, Any]: Dictionary containing duplicate detection results
        """
        # Add compatibility shim to ensure 'name' column exists
        if 'name' not in cluster_attributes.columns and 'attribute_name' in cluster_attributes.columns:
            cluster_attributes = cluster_attributes.copy()
            cluster_attributes['name'] = cluster_attributes['attribute_name']
            logger.info("Added 'name' column as alias for 'attribute_name' in cluster_attributes")
            
        # Get cluster ID for logging if available
        cluster_id = "unknown"
        if 'cluster_id' in cluster_attributes.columns and not cluster_attributes.empty:
            cluster_id = str(cluster_attributes['cluster_id'].iloc[0])
            
        logger.info(f"Starting duplicate detection for cluster {cluster_id} with {len(cluster_attributes)} attributes")
        
        # Log attribute names in the cluster
        if not cluster_attributes.empty:
            attribute_names = cluster_attributes['attribute_name'].tolist()
            logger.info(f"Attributes in cluster {cluster_id}: {attribute_names}")
        
        # Log incremental mode status
        if incremental:
            logger.info("Running in incremental mode - will prioritize existing attributes")
        
        if len(cluster_attributes) <= 1:
            # No duplicates if only one attribute in the cluster
            logger.warning(f"Cluster {cluster_id} has <= 1 attribute, no duplicates to find.")
            return {
                "attributes": cluster_attributes['attribute_name'].tolist() if not cluster_attributes.empty else [],
                "duplicates": [],
                "duplicate_groups": {}
            }
        
        else:
            # Prepare the prompt
            attributes_text = "\n".join([
                f"{i+1}. ID: {row.get('attribute_id', 'unknown_id')} | Name: {row['attribute_name']} | Definition: {row['attribute_definition']}"
                for i, (_, row) in enumerate(cluster_attributes.iterrows())
            ])
            
            prompt = f"""
            I have a set of data attributes that may contain duplicates. Analyze the following attributes and identify which ones are duplicates of each other.

            {attributes_text}

            For each attribute, determine if it is a duplicate of any other attribute in the list. Two attributes are considered duplicates if they represent the same concept or similar semantic meaning, even if they have different names or definitions.
            
            IMPORTANT: You must return your final answer as a valid JSON object EXACTLY in the following structure, with no additional text before or after the JSON:
            ```json
            {{
                "duplicates": [
                    {{
                        "id": "unique_attribute_id",
                        "name": "name_of_the_attribute",
                        "is_duplicate": a_boolean_value_of_true_or_false,
                        "group": an_integer_group_identifier_for_duplicates 
                    }},
                    ...
                ]
            }}
            ```
            Assign true to is_duplicate if the attribute is a duplicate of another attribute, and false otherwise.
            Assign the same duplicate_group_id to all attributes that are duplicates of each other. Use an integer identifier like 1, 2, 3, 4, etc. If an attribute is not a duplicate, set the group field to null (not the string "none").
            
            Here's an example of a valid response with 3 attributes where 2 are duplicates:
            ```json
            {{
                "duplicates": [
                    {{
                        "id": "attr_1",
                        "name": "customer_id",
                        "is_duplicate": true,
                        "group": 1
                    }},
                    {{
                        "id": "attr_2",
                        "name": "client_id",
                        "is_duplicate": true,
                        "group": 1
                    }},
                    {{
                        "id": "attr_3",
                        "name": "product_name",
                        "is_duplicate": false,
                        "group": null
                    }}
                ]
            }}
            ```
            
            Importantly, do not include any explanatory text before or after the JSON. Return ONLY the JSON object.
            """
            
            # Generate response using the specified provider if provided
            response = self.generate_response(prompt, provider=provider)
            
            # Save LLM response
            cluster_id_for_filename = "unknown_cluster"
            if not cluster_attributes.empty and 'cluster_id' in cluster_attributes.columns:
                cluster_id_for_filename = str(cluster_attributes['cluster_id'].iloc[0])
            self.save_response(f"detect_duplicates_cluster_{cluster_id_for_filename}", response)
            
            # Parse the JSON response using the enhanced extraction method
            logger.info("Parsing response from language model using enhanced extraction")
            result = self.extract_json_from_response(response)
            
            # Validate the parsed JSON structure
            try:        
                # test 1: check if the result is a dictionary and contains the "duplicates" key
                if not isinstance(result, dict) or "duplicates" not in result or not isinstance(result["duplicates"], list):
                    logger.error(f"Invalid JSON structure: 'duplicates' key missing or not a list. Response: {response}")
                    logger.warning("Using default empty result structure due to invalid JSON structure")
                    # Create a default result structure with no duplicates
                    result = {
                        "duplicates": []
                    }
                    # For each attribute in the cluster, add a non-duplicate entry
                    for _, row in cluster_attributes.iterrows():
                        result["duplicates"].append({
                            "id": row.get("attribute_id", "unknown_id"),
                            "name": row.get("name", row.get("attribute_name", "unknown_attribute")),
                            "is_duplicate": False
                        })
                
                # test 2: check if each item in the "duplicates" list is a dictionary and contains the required keys
                valid_items = []
                for item in result["duplicates"]:
                    if isinstance(item, dict) and \
                       "id" in item and \
                       "name" in item and \
                       "is_duplicate" in item and \
                       isinstance(item["is_duplicate"], bool) and \
                       (not item["is_duplicate"] or "group" in item):
                        valid_items.append(item)
                    else:
                        logger.error(f"Invalid item in 'duplicates' list: {item}. Skipping this item.")
                
                # Replace the duplicates list with only valid items
                result["duplicates"] = valid_items
                
                # If we ended up with no valid items, create default non-duplicate entries
                if not valid_items:
                    logger.warning("No valid items found in duplicates list. Creating default entries.")
                    for _, row in cluster_attributes.iterrows():
                        result["duplicates"].append({
                            "id": row.get("attribute_id", "unknown_id"),
                            "name": row.get("name", row.get("attribute_name", "unknown_attribute")),
                            "is_duplicate": False
                        })
                        
            except Exception as e: # Catch any errors during validation
                logger.error(f"Error during JSON validation in detect_duplicates_in_cluster: {e}. Using default structure.")
                # Create a default result structure with no duplicates
                result = {
                    "duplicates": []
                }
                # For each attribute in the cluster, add a non-duplicate entry
                for _, row in cluster_attributes.iterrows():
                    result["duplicates"].append({
                        "id": row.get("attribute_id", "unknown_id"),
                        "name": row.get("name", row.get("attribute_name", "unknown_attribute")),
                        "is_duplicate": False
                    })

            # Reformat results (integrating the logic from the previous reformat_results function)
            logger.info("Reformatting duplicate detection results")
            duplicate_groups = {}
            for item in result.get("duplicates", []):
                if item.get("is_duplicate") and "group" in item:
                    group_id = item["group"]
                    # Ensure group_id is a string, as it's used as a dictionary key
                    if not isinstance(group_id, str):
                        logger.warning(f"Duplicate group_id '{group_id}' is not a string. Converting. Item: {item}")
                        group_id = str(group_id)
                    if group_id not in duplicate_groups:
                        duplicate_groups[group_id] = []
                    duplicate_groups[group_id].append(item.get("name", "unknown_attribute"))
                        
            # Create final result dictionary for this cluster
            logger.info(f"Duplicate detection complete, found {len(duplicate_groups)} duplicate groups")
            return {
                "attributes": cluster_attributes['attribute_name'].tolist(),
                "duplicates": result.get("duplicates", []),
                "duplicate_groups": duplicate_groups
            }


    #functional process 2   
    @log_execution_time
    def selection(self, duplicate_group: List[str], attributes_df: pd.DataFrame, incremental: bool = False, provider: str = None) -> Dict[str, str]:
        """Select the best attribute from a duplicate group.
        
        Parameters:
            duplicate_group (List[str]): List of attribute names in the duplicate group
            attributes_df (pd.DataFrame): DataFrame containing attribute details (must include 'attribute_name' and 'attribute_id')
            incremental (bool, optional): Flag for incremental processing
            provider (str, optional): Model provider ('huggingface' or 'azure_openai'). If None, uses the instance's provider.
            
        Returns:
            Dict[str, str]: A dictionary containing 'best_attribute_id', 'best_attribute' (name), and 'reasoning'.
        """
        try:
            # Log duplicate group details
            logger.info(f"Selecting best attribute from group of {len(duplicate_group)} attributes")
            logger.info(f"Duplicate group: {duplicate_group}")
            
            # Log incremental mode status
            if incremental:
                logger.info("Running in incremental mode - will prioritize existing attributes")
            
            # Get attribute details
            group_df = attributes_df[attributes_df['attribute_name'].isin(duplicate_group)].copy()
            
            # Log if any attributes were not found in attributes_df
            if len(group_df) < len(duplicate_group):
                missing_attrs = set(duplicate_group) - set(group_df['attribute_name'].tolist())
                logger.warning(f"Some attributes were not found in attributes_df: {missing_attrs}")
            
            # Log attribute details for debugging
            for _, row in group_df.iterrows():
                attr_info = f"Name: {row['attribute_name']}"
                if 'attribute_definition' in row:
                    attr_info += f", Definition: {row['attribute_definition']}"
                if 'source' in row:
                    attr_info += f", Source: {row['source']}"
                if 'first_seen_in_round' in row:
                    attr_info += f", First seen in round: {row['first_seen_in_round']}"
                logger.debug(f"Attribute details: {attr_info}")
            
            # Check if we're in incremental mode and if there are any existing attributes
            if incremental and 'source' in group_df.columns and 'existing' in group_df['source'].values:
                # Prioritize existing attributes
                existing_attrs = group_df[group_df['source'] == 'existing']['attribute_name'].tolist()
                if existing_attrs:
                    logger.info(f"Found {len(existing_attrs)} existing attributes: {existing_attrs}")
                    
                    # If there are multiple existing attributes, choose the one with the lowest first_seen_in_round
                    if len(existing_attrs) > 1 and 'first_seen_in_round' in group_df.columns:
                        logger.info("Multiple existing attributes found, selecting the one with lowest first_seen_in_round")
                        # Ensure 'first_seen_in_round' is numeric for min() to work correctly if it's not already
                        group_df['first_seen_in_round'] = pd.to_numeric(group_df['first_seen_in_round'], errors='coerce')
                        earliest_round = group_df[group_df['source'] == 'existing']['first_seen_in_round'].min()
                        logger.info(f"Earliest round: {earliest_round}")
                        
                        earliest_attrs = group_df[(group_df['source'] == 'existing') & 
                                                 (group_df['first_seen_in_round'] == earliest_round)]['attribute_name'].tolist()
                        if earliest_attrs: # Ensure list is not empty after filtering
                            logger.info(f"Selected existing attribute from earliest round: {earliest_attrs[0]}")
                            return earliest_attrs[0]
                    
                    logger.info(f"Selected existing attribute: {existing_attrs[0]}")
                    return existing_attrs[0]  # Return the first existing attribute
            
            # Create text representation of attributes
            attributes_text = ""
            for _, row in group_df.iterrows():
                source_tag = f" [EXISTING]" if 'source' in row and row['source'] == 'existing' else ""
                attr_id = row.get('attribute_id', 'unknown_id')
                attributes_text += f"Attribute ID: {attr_id} | Attribute Name: {row['attribute_name']} | {source_tag} | Attribute Definition: {row['attribute_definition']}\n"
            
            prompt = f"""
            You are a data expert follows best practice of data management practice. You are given a set of data attributes that have been identified as duplicates with each other. You are tasked to analyse those data attributes and select one from name with the best quality in terms of naming convention and definition.

            This is the set of data attributes:
            {attributes_text}

            A good attribute should have:
            1. A clear, descriptive name that follows standard naming conventions
            2. A comprehensive and precise definition
            3. Consistency with industry standards or common practices
            {'4. Additional consideration is - selection priority must be given to attributes marked as [EXISTING] unless a new attribute is significantly better' if incremental else ''}

            IMPORTANT: After your analysis, you must return your final answer as a valid JSON object EXACTLY in the following structure, with no additional text before or after the JSON:
            ```json
            {{
                "best_attribute_id": "id of the best attribute",
                "best_attribute": "name of the best attribute",
                "reasoning": "explanation for why this attribute was selected"
            }}
            ```

            Here's an example of a valid response:
            ```json
            {{
                "best_attribute_id": "attr_2",
                "best_attribute": "client_id",
                "reasoning": "This attribute has a clear, descriptive name that follows standard naming conventions and has a comprehensive definition that accurately describes its purpose."
            }}
            ```
            
            Importantly, do not include any explanatory text before or after the JSON. Return ONLY the JSON object.
            """
            
            # Generate response using the specified provider if provided
            response = self.generate_response(prompt, provider=provider)

            # Save LLM response
            group_name_for_filename = "_" .join(sorted(duplicate_group))[:50] # Truncate if too long
            self.save_response(f"select_best_attribute_group_{group_name_for_filename}", response)
            
            # Parse the JSON response using the enhanced extraction method
            logger.debug(f"Attempting to extract JSON from response in select_best_attribute for group: {duplicate_group}. Response: {response[:500]}...")
            result = self.extract_json_from_response(response)

            if result and isinstance(result, dict) and "best_attribute" in result and "best_attribute_id" in result:
                best_attribute_name = result.get("best_attribute")
                best_attribute_id = result.get("best_attribute_id") # Ensure ID is fetched
                
                # Validate that the returned name is one of the candidates
                if isinstance(best_attribute_name, str) and best_attribute_name in duplicate_group:
                    logger.info(f"Successfully selected best attribute: '{best_attribute_name}' (ID: {best_attribute_id}) from group {duplicate_group} using LLM.")
                    # Return the full result dictionary
                    return result # MODIFIED: Return full dict
                else:
                    logger.error(
                        f"LLM returned 'best_attribute' ('{best_attribute_name}') but it's invalid or not in the "
                        f"duplicate_group {duplicate_group}. Response: {response[:500]}"
                    )
                    logger.warning(f"Defaulting to first attribute in group due to invalid 'best_attribute' value.")
            else:
                logger.error(
                    f"Failed to extract valid JSON with 'best_attribute' and 'best_attribute_id' keys from LLM response in select_best_attribute "
                    f"for group {duplicate_group}. Response: {response[:500]}"
                )
                logger.warning(f"Defaulting to first attribute in group due to parsing failure or missing key.")
            
            # Fallback if LLM parsing failed or attribute is invalid
            default_name = duplicate_group[0]
            # Fetch ID for the default attribute from group_df
            default_id_series = group_df[group_df['attribute_name'] == default_name]['attribute_id']
            default_id = default_id_series.iloc[0] if not default_id_series.empty else "unknown_id_fallback"
            logger.info(f"Selection defaulted to: '{default_name}' (ID: {default_id})")
            return {
                "best_attribute_id": default_id,
                "best_attribute": default_name,
                "reasoning": "Defaulted selection due to LLM response error or invalid/missing 'best_attribute' or 'best_attribute_id'."
            } # MODIFIED: Return dict for fallback
            
        except Exception as e: # Catches errors from self.generate_response() or other non-JSON issues
            logger.error(f"Error in select_best_attribute: {e}. Defaulting to first attribute.")
            if not duplicate_group:
                logger.error("Cannot select best attribute from an empty group.")
                raise ValueError("Cannot select best attribute from an empty group.")
            
            default_name = duplicate_group[0]
            default_id_series = group_df[group_df['attribute_name'] == default_name]['attribute_id']
            default_id = default_id_series.iloc[0] if not default_id_series.empty else "unknown_id_exception_fallback"
            logger.info(f"Selection defaulted (exception) to: '{default_name}' (ID: {default_id})")
            return {
                "best_attribute_id": default_id,
                "best_attribute": default_name,
                "reasoning": f"Defaulted selection due to exception: {str(e)[:100]}"
            } # MODIFIED: Return dict for fallback


@log_execution_time
def run_dedup(attributes_with_clusters: pd.DataFrame, model_path: str, 
                     output_dir: Union[str, Path] = None, provider: str = "huggingface",
                     incremental: bool = False, tmp_llm_responses_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Detect duplicates in clustered attributes.
    
    Parameters:
        attributes_with_clusters (pd.DataFrame): DataFrame containing attributes with cluster assignments
        model_path (str): Path to the language model
        output_dir (Union[str, Path]): Directory to save outputs
        provider (str): Model provider ('huggingface' or 'azure_openai')
        incremental (bool): Flag for incremental processing
        tmp_llm_responses_dir (Optional[Union[str, Path]]): Directory to save raw LLM responses.
        
    Returns:
        pd.DataFrame: DataFrame with duplicate detection results
    """
    start_time = time.time()
    logger.info("Starting duplicate detection process")
    logger.info(f"Model path: {model_path}, Provider: {provider}, Incremental: {incremental}")
    
    # Log input data statistics
    total_attributes = len(attributes_with_clusters)
    unique_clusters_count = attributes_with_clusters['cluster_id'].nunique() if 'cluster_id' in attributes_with_clusters.columns else 0
    logger.info(f"Input data: {total_attributes} attributes across {unique_clusters_count} clusters")
    
    # Log memory usage at start
    mem_start = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
    logger.info(f"Initial memory usage: {mem_start:.2f} MB")
    # ------------------------------------------------------------------
    # Compatibility shim:
    # Downstream code (DuplicateDetector.detection / .selection) expects
    # the DataFrame to have a column named 'name'.  Provide an alias to
    # 'attribute_name' if it does not already exist.
    if 'name' not in attributes_with_clusters.columns and 'attribute_name' in attributes_with_clusters.columns:
        attributes_with_clusters = attributes_with_clusters.copy()
        attributes_with_clusters['name'] = attributes_with_clusters['attribute_name']
    # ------------------------------------------------------------------
    
    try:
        # Initialize duplicate detector
        logger.info("Initializing DuplicateDetector")
        detector = DuplicateDetector(model_path, provider=provider, tmp_responses_dir=tmp_llm_responses_dir)
        
        logger.info("Loading language model")
        detector.load_model()
        
        # Create output directory if provided
        if output_dir:
            output_dir = Path(output_dir)
            logger.info(f"Creating output directory at {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        else:
            logger.info("No output directory specified, results will not be saved to disk")
        
        # Initialize result DataFrame
        logger.info("Initializing result DataFrame")
        result_df = attributes_with_clusters.copy()
        result_df['is_duplicate'] = False
        result_df['duplicate_group_id'] = pd.NA # Use pd.NA for nullable integer/string types
        result_df['keep'] = True
        result_df['best_attribute_id_for_group'] = pd.NA # NEW column
        result_df['best_attribute_reasoning_for_group'] = pd.NA # NEW column
        
        # Process each cluster
        unique_clusters = attributes_with_clusters['cluster_id'].unique()
        logger.info(f"Processing {len(unique_clusters)} clusters")
        all_duplicate_groups = {}
        
        # Track progress
        total_clusters = len(unique_clusters)
        processed_clusters = 0
        skipped_clusters = 0
        clusters_with_duplicates = 0
        total_duplicates_found = 0
        
        for cluster_id in unique_clusters:
            processed_clusters += 1
            logger.info(f"Processing cluster {cluster_id} ({processed_clusters}/{total_clusters})")
            
            # Get attributes in the cluster
            cluster_attributes = attributes_with_clusters[attributes_with_clusters['cluster_id'] == cluster_id]
            cluster_size = len(cluster_attributes)
            logger.info(f"Cluster {cluster_id} has {cluster_size} attributes")
            
            # Skip clusters with only one attribute
            if cluster_size <= 1:
                logger.info(f"Cluster {cluster_id} has only one attribute, skipping")
                skipped_clusters += 1
                continue
            
            # Detect duplicates in the cluster
            logger.info(f"Detecting duplicates in cluster {cluster_id}")
            duplicate_results = detector.detection(cluster_attributes, attributes_with_clusters, incremental=incremental)
            
            # Count duplicates found
            duplicates_in_cluster = sum(1 for item in duplicate_results.get("duplicates", []) if item.get("is_duplicate", False))
            if duplicates_in_cluster > 0:
                clusters_with_duplicates += 1
                total_duplicates_found += duplicates_in_cluster
                logger.info(f"Found {duplicates_in_cluster} duplicates in cluster {cluster_id}")
            else:
                logger.info(f"No duplicates found in cluster {cluster_id}")
            
            # Save results if output_dir is provided
            if output_dir:
                output_file = output_dir / f"cluster_{cluster_id}_duplicates.json"
                logger.debug(f"Saving cluster {cluster_id} results to {output_file}")
                with open(output_file, 'w') as f:
                    json.dump(duplicate_results, f, indent=4)
            
            # Process duplicate results and update tracking variables
            duplicate_groups_in_cluster = set()
            for item in duplicate_results.get("duplicates", []):
                attribute_id = item.get("id", "unknown_id")
                attribute_name = item.get("name", item.get("attribute_name", "unknown_attribute"))
                is_duplicate = item.get("is_duplicate", False)
                
                if is_duplicate and "group" in item:
                    group_id = item["group"]
                    # Create a unique group ID across all clusters
                    unique_group_id = f"cluster_{cluster_id}_{group_id}"
                    duplicate_groups_in_cluster.add(unique_group_id)
                    
                    # Update DataFrame - match by both ID and name for robustness
                    match_condition = ((result_df['attribute_id'] == attribute_id) | 
                                      (result_df['attribute_name'] == attribute_name))
                    result_df.loc[match_condition, 'is_duplicate'] = True
                    result_df.loc[match_condition, 'duplicate_group_id'] = unique_group_id
                    # Store the attribute ID for reference
                    result_df.loc[match_condition, 'attribute_id_ref'] = attribute_id
                    
                    # Add to all_duplicate_groups
                    if unique_group_id not in all_duplicate_groups:
                        all_duplicate_groups[unique_group_id] = []
                    all_duplicate_groups[unique_group_id].append(attribute_name)
            
            logger.info(f"Created {len(duplicate_groups_in_cluster)} duplicate groups for cluster {cluster_id}")
            
            # Log progress percentage
            progress_pct = (processed_clusters / total_clusters) * 100
            logger.info(f"Progress: {progress_pct:.1f}% ({processed_clusters}/{total_clusters} clusters processed)")
        
        logger.info(f"Cluster processing complete. Found duplicates in {clusters_with_duplicates}/{total_clusters} clusters")
        logger.info(f"Total duplicate groups identified: {len(all_duplicate_groups)}")
        logger.info(f"Total duplicate attributes identified: {total_duplicates_found}")
        
        # Select the best attribute from each duplicate group
        logger.info("Starting best attribute selection for each duplicate group")
        total_groups = len(all_duplicate_groups)
        processed_groups = 0
        
        for group_id, attribute_names in all_duplicate_groups.items():
            processed_groups += 1
            logger.info(f"Selecting best attribute for group {group_id} ({processed_groups}/{total_groups}) with attributes: {attribute_names}")
            
            if not attribute_names:
                logger.warning(f"Skipping group {group_id} as it has no attribute names listed.")
                continue

            # Call selection method which now returns a dictionary
            selection_result = detector.selection(attribute_names, attributes_with_clusters, incremental=incremental, provider=provider)
            
            best_attribute_name = selection_result.get("best_attribute")
            best_attribute_id = selection_result.get("best_attribute_id")
            reasoning = selection_result.get("reasoning")

            if not best_attribute_name:
                logger.error(f"Selection for group {group_id} did not return a 'best_attribute'. Result: {selection_result}")
                # Potentially handle this by selecting the first or skipping, here we log and continue
                continue
            
            logger.info(f"Selected best attribute for group {group_id}: '{best_attribute_name}' (ID: {best_attribute_id}) with reasoning: '{reasoning}'")
            
            # Mark the selected attribute to keep
            # Ensure we are matching on the 'attribute_name' column in result_df
            # And only for attributes belonging to the current duplicate_group_id
            group_mask = result_df['duplicate_group_id'] == group_id
            result_df.loc[group_mask, 'keep'] = False # First, mark all in group to not keep
            
            # Then, mark the selected best attribute to keep
            # Make sure best_attribute_name is valid and present for the group
            best_attr_match_in_group_mask = (result_df['attribute_name'] == best_attribute_name) & group_mask
            if best_attr_match_in_group_mask.any():
                result_df.loc[best_attr_match_in_group_mask, 'keep'] = True
                logger.info(f"Marked '{best_attribute_name}' (ID: {best_attribute_id}) to be kept for group {group_id}.")
            else:
                logger.warning(f"Best attribute '{best_attribute_name}' (ID: {best_attribute_id}) not found within group {group_id} in result_df. No attribute marked to keep for this group based on selection.")

            # Store the ID and reasoning for all attributes in this group
            if best_attribute_id:
                result_df.loc[group_mask, 'best_attribute_id_for_group'] = best_attribute_id
            if reasoning:
                result_df.loc[group_mask, 'best_attribute_reasoning_for_group'] = reasoning
            
            # Save best attribute if output_dir is provided
            if output_dir:
                output_file = output_dir / f"group_{group_id}_best_attribute.json"
                logger.debug(f"Saving best attribute selection for group {group_id} to {output_file}")
                with open(output_file, 'w') as f:
                    json.dump({
                        "group_id": group_id,
                        "attributes": attribute_names,
                        "best_attribute": best_attribute_name,
                        "best_attribute_id": best_attribute_id,
                        "reasoning": reasoning
                    }, f, indent=4)
        
        # Generate summary statistics
        total_kept = result_df['keep'].sum()
        total_removed = len(result_df) - total_kept
        logger.info(f"Duplicate detection summary:")
        logger.info(f"  - Total attributes processed: {len(result_df)}")
        logger.info(f"  - Attributes to keep: {total_kept}")
        logger.info(f"  - Duplicate attributes to remove: {total_removed}")
        logger.info(f"  - Duplicate reduction: {(total_removed/len(result_df)*100):.1f}%")
        
        # Save final results if output_dir is provided
        if output_dir:
            output_file = output_dir / "duplicate_detection_results.csv"
            logger.info(f"Saving final results to {output_file}")
            result_df.to_csv(output_file, index=False)
        
        # Log memory usage at end
        mem_end = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        mem_diff = mem_end - mem_start
        logger.info(f"Final memory usage: {mem_end:.2f} MB (change: {mem_diff:+.2f} MB)")
        
        # Log total execution time
        total_time = time.time() - start_time
        logger.info(f"Duplicate detection completed in {total_time:.2f} seconds")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error in run_dedup: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise
