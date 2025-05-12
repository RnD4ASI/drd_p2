# Data Attribute Deduplication System - Instructions

This document provides detailed instructions on how to set up and execute the data attribute deduplication system, which uses embedding models, clustering, and language models to identify and resolve duplicate data attributes.

## Table of Contents
1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Environment Variables](#environment-variables)
6. [Execution](#execution)
7. [Incremental Deduplication](#incremental-deduplication)
8. [Output](#output)
9. [Troubleshooting](#troubleshooting)

## System Overview

The deduplication system works through the following steps:

1. **Data Loading**: Load and preprocess attribute data from CSV files
   - Parse attribute IDs, names, and definitions
   - Handle incremental mode by merging with previous results

2. **Embedding**: Convert data attributes into vector representations using embedding models
   - Generate high-dimensional vectors that capture semantic meaning
   - Use either local Hugging Face models or Azure OpenAI API
   - Cache embeddings for reuse in future runs

3. **Clustering**: Group similar attributes based on their embeddings
   - Apply K-means clustering to group semantically similar attributes
   - Optimize cluster sizes to balance precision and recall
   - Generate cluster assignments for each attribute

4. **Duplicate Detection**: Analyze each cluster to identify duplicates using a language model
   - Process each cluster with an LLM to identify potential duplicates
   - Group attributes that refer to the same concept
   - Generate detailed reasoning for duplicate decisions

5. **Best Attribute Selection**: Select the best attribute from each duplicate group
   - Evaluate attributes based on naming conventions, definitions, and industry standards
   - Select one attribute to keep from each duplicate group
   - Generate reasoning for the selection decision
   - Store the attribute ID and reasoning for traceability

6. **Output Generation**: Create final results in CSV and JSON formats
   - Generate comprehensive CSV with all attributes and duplicate information
   - Include metadata about the deduplication process
   - Provide clear indicators of which attributes to keep and remove

## Prerequisites

- Python 3.8 or higher (Python 3.12 recommended)
- CUDA-compatible GPU (recommended for faster processing with local models)
- 8GB+ RAM (16GB+ recommended for large datasets)
- 10GB+ disk space for model storage
- Virtual environment management tool (venv or conda)
- Git for repository cloning

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd drd_dedup2
   ```

2. Create and activate a virtual environment:
   ```bash
   # Using venv
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # OR using conda
   conda create -n drd_dedup python=3.12
   conda activate drd_dedup
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download required models (if using local Hugging Face models):
   ```bash
   # Create model directories
   mkdir -p model
   
   # Example for downloading embedding model
   huggingface-cli download FinLang/finance-embeddings-investopedia --local-dir model/FinLang:finance-embeddings-investopedia
   
   # Example for downloading language model
   huggingface-cli download Qwen/Qwen3-1.7B --local-dir model/Qwen:Qwen3-1.7B
   ```

5. Verify installation:
   ```bash
   # Run a simple test to verify installation
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")}');"
   ```

## Configuration

The system is configured using a YAML configuration file (`config.yaml`). Here's an explanation of the configuration options:

```yaml
embedding:
  # Path to the embedding model or Azure OpenAI model name
  model_path: "model/FinLang:finance-embeddings-investopedia"
  # Provider for the embedding model: "huggingface" or "azure_openai"
  provider: "huggingface"
  # Uncomment and modify the following line to use Azure OpenAI for embeddings
  # model_path: "text-embedding-ada-002"
  # provider: "azure_openai"

clustering:
  # Maximum size of each cluster
  max_cluster_size: 10
  # Maximum number of clusters
  max_clusters: 20

language_model:
  # Path to the language model or Azure OpenAI model deployment name
  model_path: "model/Qwen:Qwen3-1.7B"
  # Provider for the language model: "huggingface" or "azure_openai"
  provider: "huggingface"
  # Uncomment and modify the following lines to use Azure OpenAI for language model
  # model_path: "gpt-4o"
  # provider: "azure_openai"

output:
  # Directory for temporary files
  tmp_dir: "tmp"
  # Directory for final results
  result_dir: "result"
```

### Configuration Options

#### Embedding Model Options

- **model_path**: 
  - For Hugging Face models: Path to the local model directory or model identifier
  - For Azure OpenAI: The deployment name of your embedding model (e.g., "text-embedding-ada-002")
- **provider**: The provider of the embedding model
  - `huggingface`: Use a local Hugging Face model
  - `azure_openai`: Use Azure OpenAI API for embeddings

#### Clustering Options

- **max_cluster_size**: Maximum number of attributes in each cluster
- **max_clusters**: Maximum number of clusters to create

#### Language Model Options

- **model_path**: 
  - For Hugging Face models: Path to the local model directory or model identifier
  - For Azure OpenAI: The deployment name of your language model (e.g., "gpt-4o")
- **provider**: The provider of the language model
  - `huggingface`: Use a local Hugging Face model
  - `azure_openai`: Use Azure OpenAI API for language model inference

#### Output Options

- **tmp_dir**: Directory for temporary files and intermediate results
- **result_dir**: Directory for final results

## Environment Variables

When using Azure OpenAI, you need to set the following environment variables:

```bash
# Create a .env file in the project root directory with the following variables
SCOPE="https://cognitiveservices.azure.com/.default"
TENANT_ID="your-tenant-id"
CLIENT_ID="your-client-id"
CLIENT_SECRET="your-client-secret"
SUBSCRIPTION_KEY="your-subscription-key"
AZURE_API_VERSION="2023-05-15"
AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com"
```

## Input Data Format

The system expects input data in CSV format with specific columns. Here are the requirements:

### Required Columns

- `attribute_id`: Unique identifier for each attribute (e.g., "attr_1", "attr_28")
- `attribute_name`: Name of the attribute (e.g., "customer_age", "years_old")
- `attribute_definition`: Description or definition of the attribute

### Optional Columns

- `source`: Source of the attribute (default: "new")
- `first_seen_in_round`: Round number when the attribute was first seen (default: current round)

### Example Input CSV

```csv
attribute_id,attribute_name,attribute_definition,source,first_seen_in_round
attr_1,customer_id,Unique identifier for the customer,new,1
attr_2,client_id,Unique identifier for the client,new,1
attr_3,user_identifier,Unique identifier for the user,new,1
attr_4,transaction_amount,Amount of the transaction in dollars,new,1
attr_5,payment_value,Value of the payment in dollars,new,1
```

### Data Preparation Guidelines

1. Ensure all attribute IDs are unique across the dataset
2. Provide clear and descriptive definitions for each attribute
3. Keep attribute names concise but descriptive
4. For incremental deduplication, maintain consistent ID formats across batches

## Execution

### Basic Execution

Run the deduplication process with the default configuration:

```bash
python -m src.main --input path/to/input.csv
```

### Using the Development Script

For development and testing, you can use the provided `dev_run.sh` script:

```bash
# Activate your virtual environment first
source .venv/bin/activate

# Run the development script
bash dev_run.sh
```

### Custom Configuration

Specify a custom configuration file:

```bash
python -m src.main --input path/to/input.csv --config custom_config.yaml
```

### Command-Line Options

- `--input`: Path to the input CSV file (required)
- `--config`: Path to the configuration file (default: `config.yaml`)
- `--run_id`: Custom run identifier (default: auto-generated from timestamp and input filename)
- `--previous_results`: Path to previous deduplication results for incremental mode
- `--incremental`: Flag to enable incremental deduplication mode

### Examples

1. Using local Hugging Face models:
   ```bash
   python -m src.main --input data/attributes.csv
   ```

2. Using Azure OpenAI for embeddings only:
   ```bash
   python -m src.main --input data/attributes.csv --embedding_provider azure_openai
   ```

3. Using Azure OpenAI for both embeddings and language model:
   ```bash
   python -m src.main --input data/attributes.csv --embedding_provider azure_openai --language_model_provider azure_openai
   ```

## Incremental Deduplication

The system supports incremental deduplication, allowing you to process new batches of attributes while maintaining consistency with previously deduplicated data.

### Overview

Incremental deduplication works in two main scenarios:

1. **Initial Deduplication (Round 1)**: Process the first batch of attributes using the standard mode.
2. **Subsequent Rounds**: Process new batches of attributes while incorporating the results from previous rounds.

### How It Works

1. The system loads the previous deduplication results and keeps only the non-duplicate attributes or the best attributes from each duplicate group.
2. New attributes are loaded and combined with the previously kept attributes.
3. The combined dataset goes through the same deduplication process (embedding, clustering, duplicate detection).
4. When selecting the best attribute from each duplicate group, the system prioritizes attributes from previous rounds.

### Command-Line Usage

```bash
python -m src.main --input new_attributes.csv --previous_results result/deduplication_results.csv --incremental
```

### Parameters

- `--input`: Path to the CSV file containing new attributes to process
- `--previous_results`: Path to the CSV file containing results from a previous deduplication round
- `--incremental`: Flag to enable incremental mode

### Example Workflow

1. **Round 1**: Process initial batch of attributes
   ```bash
   python -m src.main --input batch1.csv
   ```

2. **Round 2**: Process second batch, incorporating results from Round 1
   ```bash
   python -m src.main --input batch2.csv --previous_results result/deduplication_results.csv --incremental
   ```

3. **Round 3**: Process third batch, incorporating results from Round 2
   ```bash
   python -m src.main --input batch3.csv --previous_results result/deduplication_results_round2.csv --incremental
   ```

### Output

The output format remains the same as in standard mode, but includes additional columns:

- `source`: Indicates whether the attribute is from a previous round (`existing`) or the current round (`new`)
- `first_seen_in_round`: Indicates in which deduplication round the attribute was first processed

### Best Practices

1. **Maintain Version History**: Keep results from each deduplication round in separate files with clear naming.
2. **Consistent Configuration**: Use the same configuration settings across rounds for consistent results.
3. **Review Results**: After each round, review the results to ensure the deduplication decisions align with your expectations.

## Input Format

The input CSV file should have the following columns:
- `attribute_name`: Name of the data attribute
- `attribute_definition`: Definition or description of the data attribute

Example:
```csv
attribute_name,attribute_definition
customer_id,Unique identifier for a customer
client_id,Unique identifier assigned to each client
purchase_date,Date when the purchase was made
transaction_date,Date when the transaction occurred
```

## Output

The system generates the following outputs:

### 1. Intermediate Files

#### Embeddings
- **File**: `<output_dir>/tmp/embeddings/attribute_embeddings.npy`
- **Format**: NumPy binary file (.npy)
- **Content**: Vector embeddings for each attribute
- **Usage**: Used for clustering and can be loaded for further analysis

#### Clustered Attributes
- **File**: `<output_dir>/tmp/clustering/clustered_attributes.csv`
- **Format**: CSV
- **Columns**:
  - `attribute_name`: Name of the attribute
  - `attribute_definition`: Definition of the attribute
  - `cluster`: Cluster ID assigned to the attribute
- **Example**:
  ```csv
  attribute_name,attribute_definition,cluster
  customer_id,Unique identifier for the customer,3
  client_id,Unique identifier for the client,3
  transaction_amount,Amount of the transaction in dollars,5
  ```

#### Clustering Results
- **File**: `<output_dir>/tmp/clustering/clustering_results.json`
- **Format**: JSON
- **Content**: Detailed information about the clustering process
  - `n_clusters`: Number of clusters created
  - `cluster_sizes`: Number of attributes in each cluster
  - `inertia`: Sum of squared distances to centroids (lower is better)
  - `cluster_centers`: Centroid coordinates for each cluster
- **Example**:
  ```json
  {
    "n_clusters": 10,
    "cluster_sizes": [5, 3, 7, 2, 4, 6, 3, 2, 1, 4],
    "inertia": 42.56,
    "cluster_centers": [[0.1, 0.2, ...], ...]
  }
  ```

#### Duplicate Detection Results
- **File**: `<output_dir>/tmp/duplicates/attributes_with_duplicates.csv`
- **Format**: CSV
- **Columns**:
  - `attribute_name`: Name of the attribute
  - `attribute_definition`: Definition of the attribute
  - `cluster`: Cluster ID
  - `is_duplicate`: Whether the attribute is a duplicate
  - `duplicate_group_id`: Group ID for duplicate attributes
  - `keep`: Whether to keep this attribute (best of duplicates)
- **Example**:
  ```csv
  attribute_name,attribute_definition,cluster,is_duplicate,duplicate_group_id,keep
  transaction_amount,Amount of the transaction in dollars,5,True,cluster_5_group1,True
  payment_value,Value of the payment in dollars,5,True,cluster_5_group1,False
  ```

### 2. Final Output Files

#### Deduplication Results (CSV)
- **File**: `<output_dir>/result/deduplication_results.csv`
- **Format**: CSV
- **Columns**:
  - `attribute_name`: Original attribute name
  - `attribute_definition`: Original attribute definition
  - `is_duplicate`: Whether the attribute is a duplicate (True/False)
  - `duplicate_group_id`: Group ID for duplicate attributes (empty if not a duplicate)
  - `should_remove`: Whether the attribute should be removed (True for duplicates that are not the best attribute)
- **Example**:
  ```csv
  attribute_name,attribute_definition,is_duplicate,duplicate_group_id,should_remove
  customer_id,Unique identifier for the customer,False,,False
  transaction_amount,Amount of the transaction in dollars,True,cluster_5_group1,False
  payment_value,Value of the payment in dollars,True,cluster_5_group1,True
  ```

#### Visualization Files
- **Files**: 
  - `<output_dir>/tmp/clustering/cluster_sizes.png`: Bar chart of cluster sizes
  - `<output_dir>/tmp/clustering/cluster_visualization.png`: 2D projection of clusters (if available)
- **Format**: PNG image files
- **Content**: Visual representation of clustering results

### 3. Log Files

#### Main Log
- **File**: `deduplication.log`
- **Format**: Text file
- **Content**: Detailed log of the entire deduplication process
  - Timestamps
  - Step-by-step progress
  - Warnings and errors
  - Performance metrics
- **Example**:
  ```
  2025-05-09 15:30:12 - __main__ - INFO - Starting attribute deduplication process for data/attributes.csv
  2025-05-09 15:30:13 - __main__ - INFO - Configuration loaded: {...}
  2025-05-09 15:30:15 - __main__ - INFO - Loaded 45 attributes from data/attributes.csv
  ```

### Understanding the Output

#### Duplicate Groups

Duplicate groups are identified by their `duplicate_group_id`, which follows the pattern `cluster_X_groupY` where:
- `X` is the cluster number
- `Y` is the group number within that cluster

For example, `cluster_5_group1` indicates the first duplicate group found in cluster 5.

#### Interpreting Results

To identify which attributes to keep and which to remove:

1. Look for attributes where `is_duplicate` is `True`
2. Within each `duplicate_group_id`:
   - Keep the attribute where `should_remove` is `False`
   - Remove attributes where `should_remove` is `True`

#### Example Interpretation

From the example above:
- `transaction_amount` and `payment_value` are duplicates (same `duplicate_group_id`)
- `transaction_amount` should be kept (`should_remove` is `False`)
- `payment_value` should be removed (`should_remove` is `True`)

### Using the Results

The final deduplication results can be used to:

1. **Clean data dictionaries**: Remove duplicate attributes
2. **Standardize terminology**: Use the selected best attributes as the standard terms
3. **Improve data models**: Refactor data models to use only the non-duplicate attributes
4. **Enhance data governance**: Document the duplicate relationships for future reference

## Troubleshooting

### Common Issues

1. **Memory Errors with Local Models**:
   - Use a smaller language model
   - Reduce the batch size for embeddings
   - Use Azure OpenAI instead of local models

2. **Azure OpenAI Authentication Errors**:
   - Verify that all environment variables are set correctly
   - Check that your Azure subscription has access to the required models
   - Ensure your client credentials have the necessary permissions

3. **JSON Parsing Errors**:
   - The system includes fallback mechanisms for handling malformed JSON responses
   - Check the logs for specific error messages

4. **Performance Issues**:
   - For large datasets, consider increasing the `max_cluster_size` and `max_clusters` parameters
   - Use a GPU for faster processing with local models
   - Consider using Azure OpenAI for faster inference

### Logging

The system logs detailed information about each step of the process. You can check the log file (`deduplication.log`) for troubleshooting.

To increase the log level for more detailed information:

```python
# Modify the logging configuration in src/main.py
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('deduplication.log')
    ]
)
```

## Advanced Configuration

### Using Different Models

You can use different embedding and language models by modifying the `model_path` in the configuration file:

1. **Hugging Face Models**:
   - Use any compatible model from the Hugging Face Hub
   - Example: `model_path: "sentence-transformers/all-MiniLM-L6-v2"`

2. **Azure OpenAI Models**:
   - Use any available model in your Azure OpenAI deployment
   - For embeddings: `model_path: "text-embedding-ada-002"`
   - For language model: `model_path: "gpt-4o"` or `model_path: "gpt-35-turbo"`

### Custom Clustering

The system uses K-means clustering by default. You can modify the clustering parameters in the configuration file to adjust the clustering behavior.
