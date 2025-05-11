import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple

logger = logging.getLogger(__name__)

class DataUtility:
    """Utility class for data operations."""
    
    def __init__(self):
        """Initialize DataUtility."""
        pass
    
    def text_operation(self, operation: str, file_path: Union[str, Path], content: Any = None, file_type: str = 'txt'):
        """Perform text operations like load, save, append.
        
        Parameters:
            operation (str): Operation to perform ('load', 'save', 'append')
            file_path (Union[str, Path]): Path to the file
            content (Any): Content to save or append (if operation is 'save' or 'append')
            file_type (str): Type of file ('txt', 'json', 'csv')
            
        Returns:
            Any: Content of the file if operation is 'load', None otherwise
        """
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        if operation in ['save', 'append']:
            os.makedirs(file_path.parent, exist_ok=True)
        
        try:
            if operation == 'load':
                if file_type == 'json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
                elif file_type == 'csv':
                    return pd.read_csv(file_path)
                else:  # txt
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                    
            elif operation == 'save':
                if file_type == 'json':
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(content, f, ensure_ascii=False, indent=4)
                elif file_type == 'csv':
                    content.to_csv(file_path, index=False)
                else:  # txt
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                        
            elif operation == 'append':
                if file_type == 'json':
                    # For JSON, we need to load, update, and save
                    if file_path.exists():
                        existing_data = self.text_operation('load', file_path, file_type='json')
                        if isinstance(existing_data, list) and isinstance(content, list):
                            existing_data.extend(content)
                        elif isinstance(existing_data, dict) and isinstance(content, dict):
                            existing_data.update(content)
                        else:
                            raise ValueError("Cannot append incompatible JSON structures")
                        self.text_operation('save', file_path, existing_data, file_type='json')
                    else:
                        self.text_operation('save', file_path, content, file_type='json')
                elif file_type == 'csv':
                    if file_path.exists():
                        content.to_csv(file_path, mode='a', header=False, index=False)
                    else:
                        content.to_csv(file_path, index=False)
                else:  # txt
                    with open(file_path, 'a', encoding='utf-8') as f:
                        f.write(content)
            
            return None
        
        except Exception as e:
            logger.error(f"Error in text_operation: {e}")
            raise
    
    def format_conversion(self, data: Any, output_format: str):
        """Convert data between different formats.
        
        Parameters:
            data (Any): Data to convert
            output_format (str): Target format ('dict', 'list', 'dataframe', 'json', 'array')
            
        Returns:
            Any: Converted data
        """
        try:
            if output_format.lower() == 'dict':
                if isinstance(data, pd.DataFrame):
                    return data.to_dict(orient='records')
                elif isinstance(data, list):
                    return {i: item for i, item in enumerate(data)}
                elif isinstance(data, str):
                    return json.loads(data)
                else:
                    return dict(data)
                
            elif output_format.lower() == 'list':
                if isinstance(data, pd.DataFrame):
                    return data.values.tolist()
                elif isinstance(data, dict):
                    return list(data.values())
                elif isinstance(data, str):
                    return list(data)
                else:
                    return list(data)
                
            elif output_format.lower() == 'dataframe':
                if isinstance(data, dict):
                    return pd.DataFrame.from_dict(data, orient='index')
                elif isinstance(data, list):
                    return pd.DataFrame(data)
                elif isinstance(data, str):
                    try:
                        return pd.DataFrame(json.loads(data))
                    except:
                        return pd.DataFrame([data])
                else:
                    return pd.DataFrame(data)
                
            elif output_format.lower() == 'json':
                if isinstance(data, pd.DataFrame):
                    return data.to_json(orient='records')
                elif isinstance(data, (dict, list)):
                    return json.dumps(data)
                else:
                    return json.dumps(str(data))
                
            elif output_format.lower() == 'array':
                if isinstance(data, pd.DataFrame):
                    return data.values
                elif isinstance(data, (dict, list)):
                    return np.array(list(data.values()) if isinstance(data, dict) else data)
                else:
                    return np.array(data)
                
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
                
        except Exception as e:
            logger.error(f"Error in format_conversion: {e}")
            raise


class StatisticsUtility:
    """Utility class for statistical operations."""
    
    def __init__(self):
        """Initialize StatisticsUtility."""
        pass
    
    def set_random_seed(self, min_value: int = 0, max_value: int = 100):
        """Set a random seed.
        
        Parameters:
            min_value (int): Minimum value for random seed
            max_value (int): Maximum value for random seed
            
        Returns:
            int: Random seed
        """
        try:
            seed = np.random.randint(min_value, max_value)
            np.random.seed(seed)
            return seed
        except Exception as e:
            logger.error(f"Error in set_random_seed: {e}")
            raise


class AIUtility:
    """Utility class for AI operations."""
    
    def __init__(self):
        """Initialize AIUtility."""
        pass
    
    def process_token(self, text: str, operation: str = "count"):
        """Process tokens in text.
        
        Parameters:
            text (str): Text to process
            operation (str): Operation to perform ('count')
            
        Returns:
            int: Token count if operation is 'count'
        """
        try:
            if operation == "count":
                # Simple estimation: 4 characters per token
                return len(text) // 4
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.error(f"Error in process_token: {e}")
            raise
