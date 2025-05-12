import os
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import json
import yaml
import time
from datetime import datetime
from dotenv import load_dotenv

from src.utility import DataUtility
from src.embedding import run_embedding
from src.clustering import run_clustering
from src.deduping import run_dedup

# Configure logging - basic setup for initial startup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Will be updated with file handler for each run

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or use default configuration.
    
    Parameters:
        config_path (Optional[str]): Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    default_config = {
        "embedding": {
            "model_path": "model/finance_embeddings",
            "provider": "huggingface"
        },
        "clustering": {
            "max_cluster_size": 10,
            "max_clusters": 50
        },
        "language_model": {
            "model_path": "model/qwen3-1.7B",
            "provider": "huggingface"
        },
        "output": {
            "tmp_dir": "tmp",
            "result_dir": "result"
        }
    }
    
    if config_path:
        try:
            config_path = Path(config_path)
            if config_path.suffix == '.json':
                with open(config_path, 'r') as f:
                    config = json.load(f)
            elif config_path.suffix in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                logger.warning(f"Unsupported config file format: {config_path.suffix}")
                config = default_config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            config = default_config
    else:
        config = default_config
    
    return config

def load_previous_embeddings(embedding_dir: Path, attribute_ids: List) -> Optional[np.ndarray]:
    """Load the most recent embeddings from the embedding directory if available.
    
    Parameters:
        embedding_dir (Path): Directory containing embedding files
        attribute_ids (List): List of attribute IDs to match with saved embeddings
        
    Returns:
        Optional[np.ndarray]: Loaded embeddings if available and matching, None otherwise
    """
    try:
        # Check if the directory exists
        if not embedding_dir.exists():
            logger.info(f"Embedding directory {embedding_dir} does not exist")
            return None
            
        # Find all embedding files (sorted by timestamp, newest first)
        embedding_files = sorted(
            [f for f in embedding_dir.glob("embeddings_*.npy") if f.stem.endswith("_metadata") is False],
            key=lambda f: int(f.stem.split("_")[1]),
            reverse=True
        )
        
        if not embedding_files:
            logger.info("No embedding files found")
            return None
            
        # Try to load the most recent embedding file
        for embedding_file in embedding_files:
            try:
                from src.embedding import load_embeddings
                embeddings, metadata = load_embeddings(embedding_file)
                
                # Check if the attribute IDs match
                if metadata and "attribute_ids" in metadata:
                    saved_ids = set(metadata["attribute_ids"])
                    current_ids = set(attribute_ids)
                    
                    # If all current IDs are in the saved IDs, we can use these embeddings
                    if current_ids.issubset(saved_ids):
                        logger.info(f"Found matching embeddings in {embedding_file}")
                        
                        # If the sets are identical, return as is
                        if saved_ids == current_ids:
                            return embeddings
                            
                        # Otherwise, we need to filter the embeddings to match the current IDs
                        else:
                            # Create a mapping from ID to index
                            id_to_index = {id: i for i, id in enumerate(metadata["attribute_ids"])}
                            
                            # Get the indices of the current IDs
                            indices = [id_to_index[id] for id in attribute_ids if id in id_to_index]
                            
                            # Filter the embeddings
                            filtered_embeddings = embeddings[indices]
                            logger.info(f"Filtered embeddings from {len(embeddings)} to {len(filtered_embeddings)}")
                            return filtered_embeddings
            except Exception as e:
                logger.warning(f"Failed to load embeddings from {embedding_file}: {e}")
                continue
                
        logger.info("No matching embeddings found")
        return None
    except Exception as e:
        logger.warning(f"Error loading previous embeddings: {e}")
        return None

def process_attributes(input_file: str, config_path: Optional[str] = None, previous_results: Optional[str] = None, incremental: bool = False, run_id: Optional[str] = None) -> str:
    logger.info("========== STARTING ATTRIBUTE PROCESSING ==========")
    """Process data attributes to identify duplicates.
    
    Parameters:
        input_file (str): Path to the input CSV file
        config_path (Optional[str]): Path to the configuration file
        previous_results (Optional[str]): Path to previous deduplication results (for incremental mode)
        incremental (bool): Whether to run in incremental mode
        run_id (Optional[str]): Custom run ID for the deduplication process
        
    Returns:
        str: Path to the output CSV file
    """
    start_time = time.time()
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(config_path)
    logger.info(f"Configuration loaded: Using {config['language_model']['provider']} model {config['language_model']['model_path']}")
    
    # Generate a unique run ID if not provided
    if not run_id:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = Path(input_file).stem
        run_id = f"{timestamp}_{batch_name}"
    
    # Create organized output directories
    logger.info("Setting up directory structure...")
    base_tmp_dir = Path(config["output"]["tmp_dir"])
    base_result_dir = Path(config["output"]["result_dir"])
    
    # Create run-specific directories
    tmp_dir = base_tmp_dir / run_id
    result_dir = base_result_dir / run_id
    
    # Create all necessary directories
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    # Create subdirectories for organization
    embedding_dir = tmp_dir / "embeddings"
    clustering_dir = tmp_dir / "clustering"
    duplicate_dir = tmp_dir / "duplicates"
    llm_responses_dir = tmp_dir / "llm_responses"
    os.makedirs(embedding_dir, exist_ok=True)
    os.makedirs(clustering_dir, exist_ok=True)
    os.makedirs(duplicate_dir, exist_ok=True)
    os.makedirs(llm_responses_dir, exist_ok=True)
    
    # Set up logging for this run
    log_file = result_dir / "deduplication.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log the start of the run
    logger.info(f"Starting deduplication run: {run_id}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Mode: {'Incremental' if incremental else 'Standard'}")
    
    # Make sure to remove the file handler at the end of the function
    try:
        logger.info("Initializing data utility...")
        data_utility = DataUtility()
        
        # Handle incremental mode
        if incremental and previous_results:
            # Load previous deduplication results
            logger.info(f"Loading previous deduplication results from {previous_results}")
            logger.info("PROGRESS: 10% - Loading previous results")
            previous_df = data_utility.text_operation('load', previous_results, file_type='csv')
            
            # Keep only non-duplicate or best attributes from previous results
            previous_keep_df = previous_df[~previous_df['should_remove']].copy()
            previous_keep_df['source'] = 'existing'
            previous_keep_df['first_seen_in_round'] = 1 if 'first_seen_in_round' not in previous_keep_df.columns else previous_keep_df['first_seen_in_round']
            logger.info(f"Kept {len(previous_keep_df)} attributes from previous results")
            
            # Load new attributes
            logger.info(f"Loading new attributes from {input_file}")
            new_attributes_df = data_utility.text_operation('load', input_file, file_type='csv')
            new_attributes_df['source'] = 'new'
            new_attributes_df['first_seen_in_round'] = 2  # Mark as seen in round 2
            logger.info(f"Loaded {len(new_attributes_df)} new attributes from {input_file}")
            logger.info("PROGRESS: 15% - Loaded new attributes")
            
            # Combine previous and new attributes
            attributes_df = pd.concat([previous_keep_df, new_attributes_df], ignore_index=True)
            attributes_df['is_duplicate'] = False  # Reset duplicate status
            attributes_df['group'] = None  # Reset group ID
            attributes_df['should_remove'] = False  # Reset removal flag
            
            logger.info(f"Combined dataset contains {len(attributes_df)} attributes")
        else:
            # Standard mode - load new attributes only
            logger.info(f"Loading attributes from {input_file} (standard mode)")
            attributes_df = data_utility.text_operation('load', input_file, file_type='csv')
            attributes_df['source'] = 'new'
            attributes_df['first_seen_in_round'] = 1  # First round
            logger.info(f"Loaded {len(attributes_df)} attributes from {input_file}")
            logger.info("PROGRESS: 15% - Loaded attributes")
        
        # Save a copy of the original data
        logger.info("Saving copy of original data...")
        attributes_df.to_csv(tmp_dir / "original_attributes.csv", index=False)
        logger.info("PROGRESS: 20% - Saved original data")
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        raise
    
    # Step 1: Apply text embedding model
    try:
        logger.info("Step 1: Applying text embedding model")
        logger.info("PROGRESS: 25% - Starting embedding generation")
        
        # Check if we have attribute_id column in the dataframe
        if 'attribute_id' in attributes_df.columns:
            # Try to load existing embeddings first
            logger.info("Checking for existing embeddings...")
            existing_embeddings = load_previous_embeddings(embedding_dir, attributes_df['attribute_id'].tolist())
            
            if existing_embeddings is not None:
                logger.info(f"Using existing embeddings with shape {existing_embeddings.shape}")
                embeddings = existing_embeddings
                
                # Save a copy in the standard location for backward compatibility
                logger.info("Saving existing embeddings to standard location for compatibility...")
                np.save(embedding_dir / "name_embeddings.npy", embeddings)
                logger.info("PROGRESS: 40% - Existing embeddings loaded and saved")
            else:
                # No matching embeddings found, generate new ones
                logger.info("No matching embeddings found, generating new ones...")
                
                # Create metadata for embeddings
                embedding_metadata = {
                    "run_id": run_id,
                    "input_file": input_file,
                    "mode": "incremental" if incremental else "standard",
                    "timestamp": datetime.now().isoformat(),
                    "config": {
                        "embedding": config["embedding"],
                        "clustering": config["clustering"]
                    }
                }
                
                # Generate embeddings and save them with full precision
                embeddings = run_embedding(
                    attributes_df=attributes_df,
                    model_path=config["embedding"]["model_path"],
                    provider=config["embedding"]["provider"],
                    save_to_dir=embedding_dir,  # Save to the embedding directory
                    metadata=embedding_metadata  # Include metadata
                )
                logger.info(f"Generated {len(embeddings)} embeddings")
                
                # Also save a copy in the standard location for backward compatibility
                logger.info("Saving embeddings to standard location for compatibility...")
                np.save(embedding_dir / "name_embeddings.npy", embeddings)
                logger.info("PROGRESS: 40% - Embeddings generated and saved with full precision")
        else:
            # No attribute_id column, can't match with existing embeddings
            logger.info("No attribute_id column found, generating new embeddings...")
            
            # Create metadata for embeddings
            embedding_metadata = {
                "run_id": run_id,
                "input_file": input_file,
                "mode": "incremental" if incremental else "standard",
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "embedding": config["embedding"],
                    "clustering": config["clustering"]
                }
            }
            
            # Generate embeddings and save them with full precision
            embeddings = run_embedding(
                attributes_df=attributes_df,
                model_path=config["embedding"]["model_path"],
                provider=config["embedding"]["provider"],
                save_to_dir=embedding_dir,  # Save to the embedding directory
                metadata=embedding_metadata  # Include metadata
            )
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Also save a copy in the standard location for backward compatibility
            logger.info("Saving embeddings to standard location for compatibility...")
            np.save(embedding_dir / "name_embeddings.npy", embeddings)
            logger.info("PROGRESS: 40% - Embeddings generated and saved with full precision")
    except Exception as e:
        logger.error(f"Error in embedding step: {e}")
        raise
    
    # Step 2: Apply K-means clustering
    try:
        logger.info("Step 2: Applying K-means clustering")
        logger.info("PROGRESS: 45% - Starting clustering")
        attributes_with_clusters, clustering_results = run_clustering(
            embeddings=embeddings,
            attributes_df=attributes_df,
            max_cluster_size=config["clustering"]["max_cluster_size"],
            max_clusters=config["clustering"]["max_clusters"],
            output_dir=clustering_dir
        )
        logger.info(f"Clustered attributes into {len(attributes_with_clusters['cluster_id'].unique())} clusters")
        logger.info("PROGRESS: 60% - Clustering complete")
        
        # Save clustering results
        with open(clustering_dir / "clustering_results.json", 'w') as f:
            # Convert numpy arrays and numpy data types to Python native types for JSON serialization
            serializable_results = {}
            for k, v in clustering_results.items():
                if isinstance(v, np.ndarray):
                    serializable_results[k] = v.tolist()
                elif isinstance(v, (np.int32, np.int64)):
                    serializable_results[k] = int(v)
                elif isinstance(v, (np.float32, np.float64)):
                    serializable_results[k] = float(v)
                else:
                    serializable_results[k] = v
            json.dump(serializable_results, f, indent=4)
    except Exception as e:
        logger.error(f"Error in clustering step: {e}")
        raise
    
    # Step 3 & 4: Detect duplicates and select best attributes
    try:
        logger.info("Step 3 & 4: Detecting duplicates and selecting best attributes")
        logger.info("PROGRESS: 65% - Starting duplicate detection")
        result_df = run_dedup(
            attributes_with_clusters=attributes_with_clusters,
            model_path=config["language_model"]["model_path"],
            output_dir=duplicate_dir,
            provider=config["language_model"]["provider"],
            incremental=incremental,
            tmp_llm_responses_dir=llm_responses_dir
        )
        logger.info(f"Completed duplicate detection")
        
        # Prepare final output DataFrame
        # Ensure all relevant columns from result_df are present in final_df
        final_df = result_df.copy() 
        
        # Define columns for the final CSV output to ensure consistency
        # Make sure to include the new columns
        final_columns = [
            'attribute_id', 'attribute_name', 'attribute_definition',
            'cluster_id', 'is_duplicate', 'duplicate_group_id',
            'best_attribute_id_for_group', 'best_attribute_reasoning_for_group', # New columns
            'keep', 'source', 'first_seen_in_round'
        ]
        
        # Add any missing columns to final_df with default values (e.g., pd.NA)
        for col in final_columns:
            if col not in final_df.columns:
                logger.warning(f"Column '{col}' not found in result_df. Adding it to final_df with NA values.")
                final_df[col] = pd.NA
                
        # Select and reorder columns for the final output
        final_df = final_df[final_columns]
        
        # Save final results
        final_output_path = result_dir / "deduplication_results.csv"
        logger.info(f"Saving final deduplication results to {final_output_path}")
        data_utility.text_operation('save', str(final_output_path), content=final_df, file_type='csv')
        logger.info("PROGRESS: 90% - Final results saved")

        # Calculate and save run metadata
        run_metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "input_file": str(Path(input_file).resolve()),
            "output_directory": str(result_dir.resolve()),
            "log_file": str(log_file.resolve()),
            "config": config,
            "incremental_mode": incremental,
            "previous_results_file": str(Path(previous_results).resolve()) if previous_results else None,
            "total_attributes_processed": len(attributes_df),
            "total_attributes_in_final_output": len(final_df),
            "attributes_kept": int(final_df['keep'].sum()),
            "attributes_marked_for_removal (duplicates)": int((~final_df['keep']).sum()),
            "unique_clusters_identified_by_vector_embedding": int(final_df['cluster_id'].nunique() if 'cluster_id' in final_df.columns else 0),
            "duplicate_groups_identified_by_llm": int(final_df['duplicate_group_id'].nunique() if 'duplicate_group_id' in final_df.columns else 0),
            "execution_time_seconds": round(time.time() - start_time, 2),
            "final_csv_path": str(final_output_path.resolve())
        }
        
        # Save metadata
        with open(result_dir / "run_metadata.json", 'w') as f:
            json.dump(run_metadata, f, indent=4)
        logger.info("PROGRESS: 95% - Run metadata saved")
        
        # Also save a copy to the main results directory for easy access
        # This is especially useful for incremental mode to find the latest results
        main_output_file = Path(config["output"]["result_dir"]) / "latest_deduplication_results.csv"
        final_df.to_csv(main_output_file, index=False)
        
        # Save summary (using stats from run_metadata)
        with open(result_dir / "deduplication_summary.json", 'w') as f:
            json.dump(run_metadata, f, indent=4)
        # Log completion
        logger.info(f"Deduplication completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Found {run_metadata['attributes_marked_for_removal (duplicates)']} duplicates in {run_metadata['total_attributes_processed']} attributes")
        logger.info(f"Final results saved to {final_output_path}")
        logger.info("========== ATTRIBUTE PROCESSING COMPLETE ===========")
        
        return str(final_output_path)
    except Exception as e:
        logger.error(f"Error in final output generation: {e}")
        raise
    finally:
        # Clean up the file handler
        logger.removeHandler(file_handler)
        file_handler.close()

def main():
    """Main function to run the attribute deduplication process."""
    parser = argparse.ArgumentParser(description="Data attribute deduplication")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--previous_results", type=str, help="Path to previous deduplication results for incremental mode")
    parser.add_argument("--incremental", action="store_true", help="Run in incremental mode, building on previous results")
    parser.add_argument("--run_id", type=str, help="Custom run identifier (default: auto-generated from timestamp and batch name)")
    args = parser.parse_args()
    
    # Validate arguments for incremental mode
    if args.incremental and not args.previous_results:
        parser.error("--previous_results is required when using --incremental mode")
    
    try:
        output_file = process_attributes(
            args.input, 
            args.config, 
            args.previous_results if args.incremental else None,
            args.incremental,
            args.run_id
        )
        
        print(f"Deduplication complete. Results saved to {output_file}")
        return 0
    except Exception as e:
        logging.error(f"Error in deduplication process: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1  # Return non-zero exit code to signal failure

if __name__ == "__main__":
    exit(main())
