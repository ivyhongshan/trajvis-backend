#!/bin/bash
set -e

PORT=8081
IMAGE="us-central1-docker.pkg.dev/in-iusm-bhds-depot/cloud-run-source-deploy/trajvis-backend/hliu-trajvis-backend:085ebb2c2f61099e784342b262e4035a5ab29b42"

echo ">>> Running container on port $PORT..."
docker run --rm -d -p $PORT:8080 --name test-preload-umap $IMAGE

echo ">>> Wait 20s for preload thread to run..."
sleep 20

echo ">>> Testing first /api/umap request"
curl -w "\nTime total: %{time_total}s\n" -o /dev/null -s "http://localhost:$PORT/api/umap"

echo ">>> Testing second /api/umap request"
curl -w "\nTime total: %{time_total}s\n" -o /dev/null -s "http://localhost:$PORT/api/umap"

echo ">>> Stopping container..."
docker stop test-preload-umap
