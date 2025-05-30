#!/bin/bash

docker compose -f docker-compose.yml up -d
docker exec -it ollama ollama create my-phi3.5  -f /modelfiles/my-phi3.5/Modelfile
docker exec -it ollama ollama create my-phi4-mini  -f /modelfiles/my-phi4-mini/Modelfile
docker exec -it ollama ollama pull phi4-mini
