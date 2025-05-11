#!/bin/bash
# Dev dataset run for deduplication pipeline
cd "$(dirname "$0")"

# Check if Azure OpenAI is actually being used (not commented out)
AZURE_USED=false

# Check if there's an uncommented line with azure_openai in the config
if grep -v "^\s*#" config.yaml | grep -q "provider:\s*\"azure_openai\""; then
    AZURE_USED=true
fi

# If Azure is used, check for .env file
if [ "$AZURE_USED" = true ] && [ ! -f ".env" ]; then
    echo "[DEV RUN] Error: .env file not found but Azure OpenAI is configured in config.yaml"
    echo "[DEV RUN] Please create a .env file with Azure OpenAI credentials (copy from env template)"
    exit 1
fi

# Create directories if they don't exist
mkdir -p result
mkdir -p tmp

echo "======================================================="
echo "Data Attribute Deduplication System - DEV RUN"
echo "======================================================="
echo ""

# Set timestamp for this test run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "[DEV RUN] Test run started at: $(date)"
echo "[DEV RUN] Using provider: $(if [ "$AZURE_USED" = true ]; then echo "azure_openai"; else echo "huggingface"; fi)"
echo ""

# Define dev datasets - using only CSV files as input
DEV_FILES=(data/dev1_with_id.csv)

for FILE in "${DEV_FILES[@]}"; do
    echo "[DEV RUN] Running deduplication on $FILE..."
    python -m src.main --input "$FILE" --run_id "${TIMESTAMP}_$(basename $FILE .csv)_dev"
    if [ $? -ne 0 ]; then
        echo "[DEV RUN] Error: Deduplication failed for $FILE. Exiting."
        exit 1
    fi
    echo "[DEV RUN] Deduplication completed for $FILE."
done

echo ""
echo "[DEV RUN] All dev runs completed successfully."
echo "[DEV RUN] Results saved to result/${TIMESTAMP}_*_dev/ directories"
echo ""

# Print summary
echo "======================================================="
echo "DEV Run Summary"
echo "======================================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Provider: $(if [ "$AZURE_USED" = true ]; then echo "azure_openai"; else echo "huggingface"; fi)"
echo ""
echo "To view detailed results, check the run_metadata.json and deduplication.log files in each result directory."
echo "======================================================="
