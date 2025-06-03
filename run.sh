#!/bin/bash

docker compose -f docker-compose-no-gpu.yml up -d
sleep 3
docker exec -it ollama ollama pull phi4-mini
docker exec -it ollama ollama pull gemma3
docker exec -it ollama ollama pull qwen3:4b
# docker exec -it ollama ollama create my-phi3.5  -f modelfiles/my-phi3.5/Modelfile
# docker exec -it ollama ollama create my-phi4-mini  -f modelfiles/my-phi4-mini/Modelfile
