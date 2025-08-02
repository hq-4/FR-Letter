# Federal Register Monitoring & Summarization System

A sophisticated pipeline that monitors the Federal Register, analyzes regulatory documents using AI, and publishes summaries to multiple channels.

## Features

- **Data Ingestion**: Retrieves daily entries from Federal Register API with filtering
- **Impact Scoring**: Configurable weights for agency importance and document characteristics
- **AI Processing**: Local Ollama embeddings and summarization + OpenRouter for final summaries
- **Vector Storage**: Redis with RediSearch for similarity-based document ranking
- **Multi-channel Publishing**: Substack and Telegram integration
- **Orchestration**: Automated daily pipeline runs with monitoring

## Architecture

```
Federal Register API → Impact Scoring → Embedding (Ollama) → Vector Storage (Redis)
                                                                      ↓
Substack + Telegram ← Final Summaries (OpenRouter) ← Local Summaries (Ollama)
```

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start Local Services**
   ```bash
   # Start Redis with RediSearch
   docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
   
   # Install and start Ollama
   curl -fsSL https://ollama.ai/install.sh | sh
   ollama pull qwen2:1.5b  # For embeddings
   ollama pull mistral:latest  # For chunk summaries
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run Pipeline**
   ```bash
   python -m fr_monitor.main
   ```

## Configuration

- **Impact Scoring**: Adjust weights in `config/scoring.yaml`
- **System Prompts**: Modify `config/prompts/politico_style.txt`
- **Pipeline Schedule**: Configure in `config/pipeline.yaml`

## API Usage Limits

- OpenRouter: Maximum 5 calls per day
- Pipeline Runtime: Target <5 minutes end-to-end
- Monthly Cost: <$5 for external APIs

## Monitoring

- Logs: `logs/pipeline.log`
- Metrics: Available via Prefect dashboard
- Alerts: Configured for pipeline failures and API errors
