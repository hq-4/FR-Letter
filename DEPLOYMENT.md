# Deployment Guide: Host Ollama Setup

## Overview
This setup uses your host's existing Ollama instance instead of running it in Docker, resolving port conflicts.

## Quick Start

### 1. Start Services
```bash
# Start only Redis (Ollama is already running on host)
docker-compose up -d redis

# Check status
docker-compose ps
```

### 2. Configure Environment
Copy `.env.example` to `.env` and update:
```bash
cp .env.example .env
```

Key settings for host Ollama:
- `OLLAMA_HOST=http://localhost:11434` (or your host IP)
- `REDIS_HOST=localhost` (if running locally)
- `REDIS_PASSWORD=` (leave blank if no auth)

### 3. Verify Setup
```bash
# Check Redis is running
docker ps | grep redis

# Check Ollama is accessible
curl http://localhost:11434/api/tags

# Test the Federal Register client
python -m fr_monitor.cli test-ingestion
```

### 4. Pull Required Models
```bash
# Ensure models are available on host Ollama
ollama pull qwen2:1.5b
ollama pull mistral:latest
```

## Troubleshooting

### Port Conflicts
If Redis port 6379 is also in use:
1. Stop the conflicting service: `sudo systemctl stop redis`
2. Or change the port mapping in docker-compose.yml:
   ```yaml
   ports:
     - "6380:6379"  # Map to different host port
   ```
3. Update `REDIS_PORT=6380` in .env

### Network Issues
If Ollama isn't accessible from containers:
1. Use `host.docker.internal` instead of `localhost`:
   ```bash
   OLLAMA_HOST=http://host.docker.internal:11434
   ```
2. Or use your machine's IP address:
   ```bash
   OLLAMA_HOST=http://192.168.1.100:11434
   ```

### Verification Commands
```bash
# Test Redis connection
docker exec fr-monitor-redis redis-cli ping

# Test Ollama models
curl http://localhost:11434/api/tags | jq '.models[].name'

# Test full pipeline
docker-compose up -d redis
python -m fr_monitor.cli run --dry-run
```

## Production Notes
- Use `host.docker.internal` for container-to-host communication
- Ensure firewall allows connections to Ollama port 11434
- Consider using a reverse proxy for production deployments
