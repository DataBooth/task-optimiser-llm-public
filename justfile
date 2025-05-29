default:
    @just --list


docker-desktop:
    open /Applications/Docker.app/

ollama-docker-local:
    @echo "Running Ollama Docker image locally..."
    docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama


ollama-local:
    @echo "Opening Ollama Desktop app..."
    open /Applications/Ollama.app/


ollama-docker-local-llama2:
    docker exec -it ollama ollama pull llama2

test-ollama-docker-local-llama2:
    @echo "Testing Ollama Docker image with Llama2..."
    curl -X POST http://localhost:11434/api/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "llama2", "prompt": "What is the capital of Australia?"}'
    @echo "Test completed."