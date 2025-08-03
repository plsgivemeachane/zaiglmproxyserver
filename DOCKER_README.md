# GLM OpenAI Server - Docker Deployment

This directory contains Docker configuration for running the OpenAI-compatible GLM server.

## Prerequisites

- Docker and Docker Compose installed
- GLM API token from Zhipu AI

## Quick Start

### 1. Environment Setup

Create a `.env` file in the project root:

```bash
GLM_API_TOKEN=your_glm_api_token_here
PORT=5000
```

### 2. Using Docker Compose (Recommended)

```bash
# Build and start the service
docker-compose up -d

# View logs
docker-compose logs -f glm-server

# Stop the service
docker-compose down
```

### 3. Using Docker directly

```bash
# Build the image
docker build -t openai-glm-server .

# Run the container
docker run -d \
  --name glm-server \
  -p 5000:5000 \
  -e GLM_API_TOKEN=your_token_here \
  -v $(pwd)/logs:/app/logs \
  openai-glm-server
```

## API Endpoints

Once running, the server provides OpenAI-compatible endpoints:

### V1 API (with thinking tags)
- `POST /v1/chat/completions` - Chat completions with thinking tags
- `GET /v1/models` - List available models

### V2 API (answer only, no tags)
- `POST /v2/chat/completions` - Chat completions (answer only, no thinking tags)
- `GET /v2/models` - List available models for v2 API

### Other endpoints
- `GET /health` - Health check
- `POST /conversation/save` - Save conversation
- `POST /conversation/load` - Load conversation

## Testing

```bash
# Health check
curl http://localhost:5000/health

# V1 Chat completion (with thinking tags)
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4-plus",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'

# V2 Chat completion (answer only, no tags)
curl -X POST http://localhost:5000/v2/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4-plus",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'

# List models (V1)
curl http://localhost:5000/v1/models

# List models (V2)
curl http://localhost:5000/v2/models
```

## Configuration

### Environment Variables

- `GLM_API_TOKEN` - Your GLM API token (required)
- `PORT` - Server port (default: 5000)

### Volumes

- `./logs:/app/logs` - Server logs
- `./conversations:/app/conversations` - Saved conversations

## Troubleshooting

### Check container status
```bash
docker-compose ps
```

### View logs
```bash
docker-compose logs glm-server
```

### Restart service
```bash
docker-compose restart glm-server
```

### Access container shell
```bash
docker-compose exec glm-server /bin/bash
```

## Security Notes

- Never commit your `.env` file with real API tokens
- Use environment variables or Docker secrets for production
- Consider using a reverse proxy (nginx) for production deployments
- The server runs on `0.0.0.0:5000` inside the container for Docker networking

## Production Deployment

For production, consider:

1. Using a multi-stage Docker build
2. Running as non-root user
3. Setting up proper logging and monitoring
4. Using Docker secrets for sensitive data
5. Implementing rate limiting and authentication