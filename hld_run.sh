#!/bin/bash
# Holdout dataset run for deduplication pipeline
cd "$(dirname "$0")"

# Check if Azure OpenAI is actually being used (not commented out)
AZURE_USED=false

# Check if there's an uncommented line with azure_openai in the config
if grep -v "^\s*#" config.yaml | grep -q "provider:\s*\"azure_openai\""; then
    AZURE_USED=true
fi

# If Azure is used, check for .env file
if [ "$AZURE_USED" = true ] && [ ! -f ".env" ]; then
    echo "[HLD RUN] Error: .env file not found but Azure OpenAI is configured in config.yaml"
    echo "[HLD RUN] Please create a .env file with Azure OpenAI credentials (copy from env template)"
    exit 1
fi

# Create directories if they don't exist
mkdir -p result
mkdir -p tmp

echo "======================================================="
echo "Data Attribute Deduplication System - HOLDOUT RUN"
echo "======================================================="
echo ""

# Set timestamp for this test run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "[HLD RUN] Test run started at: $(date)"
echo "[HLD RUN] Using provider: $(if [ "$AZURE_USED" = true ]; then echo "azure_openai"; else echo "huggingface"; fi)"
echo ""

# Standard run with hld1.csv
echo "[HLD RUN] Running standard deduplication on data/hld1.csv..."
python -m src.main --input data/hld1.csv --run_id "${TIMESTAMP}_hld1_standard"

# Check if the run was successful
if [ $? -ne 0 ]; then
    echo "[HLD RUN] Error: Standard deduplication failed for hld1.csv. Exiting."
    exit 1
fi
echo "[HLD RUN] Standard deduplication completed for hld1.csv."
echo ""

# Save timestamp for incremental run
PRIOR_TIMESTAMP=${TIMESTAMP}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "[HLD RUN] Incremental run started at: $(date)"
echo ""

# Incremental run with hld2.csv
echo "[HLD RUN] Running incremental deduplication on data/hld2.csv..."
python -m src.main --input data/hld2.csv --previous_results result/${PRIOR_TIMESTAMP}_hld1_standard/deduplication_results.csv --incremental --run_id "${TIMESTAMP}_hld2_incremental"

# Check if the run was successful
if [ $? -ne 0 ]; then
    echo "[HLD RUN] Error: Incremental deduplication failed for hld2.csv. Exiting."
    exit 1
fi
echo "[HLD RUN] Incremental deduplication completed for hld2.csv."

echo ""
echo "[HLD RUN] All holdout runs completed successfully."
echo "[HLD RUN] Results saved to result/${TIMESTAMP}_*_hld/ directories"
echo ""

# Print summary
echo "======================================================="
echo "HOLDOUT Run Summary"
echo "======================================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Provider: $(if [ "$AZURE_USED" = true ]; then echo "azure_openai"; else echo "huggingface"; fi)"
echo ""
echo "To view detailed results, check the run_metadata.json and deduplication.log files in each result directory."
echo "======================================================="
