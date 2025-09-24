#!/bin/bash
set -e

BUCKET=gs://trajvis-data-20250923

echo "Uploading artifacts..."
gsutil -m cp data/artifacts/* $BUCKET/data/artifacts/

echo "Uploading CSVs..."
gsutil -m cp data/*.csv $BUCKET/data/

echo "Uploading NPYs..."
gsutil -m cp data/*.npy $BUCKET/data/

echo "? All files uploaded to $BUCKET"
