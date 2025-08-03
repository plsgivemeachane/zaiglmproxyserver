# GLM OpenAI-Compatible API Wrapper

A Python server that provides OpenAI-compatible API endpoints for GLM (ChatGLM) models from Zhipu AI. This wrapper allows you to use GLM models with any OpenAI-compatible client or application.

## Features

- **OpenAI-Compatible API** - Drop-in replacement for OpenAI API endpoints
- **Two API Versions**:
  - **v1**: Returns responses with thinking tags (`<think>...</think>`)
  - **v2**: Returns clean responses without thinking tags
- **Streaming Support** - Real-time response streaming
- **Multiple GLM Models** - Support for various GLM model variants
- **Docker Support** - Easy deployment with Docker and Docker Compose
- **Conversation Management** - Save and load conversation history

## Installation

### Option 1: Local Setup

1. Clone this repository:
```bash
git clone https://github.com/plsgivemeachane/zaiglmproxyserver
cd zaiglmproxyserver
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables (see Configuration section below)

### Option 2: Docker Setup

See [DOCKER_README.md](DOCKER_README.md) for detailed Docker deployment instructions.

## Configuration

### Getting Your GLM API Token

1. Go to the ZAI platform website: https://chat.z.ai/
2. Log in to your account
3. Open your browser's Developer Tools (F12)
4. Go to the **Application** tab
5. In the left sidebar, expand **Local Storage**
6. Click on the ZAI website domain
7. Look for a key named `token` or similar
8. Copy the token value

### Environment Setup

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your GLM API token:
```bash
GLM_API_TOKEN=your_glm_api_token_here
PORT=8976
```

## Usage

_0727-360B-API means GLM 4.5_

### Starting the Server

```bash
python openai_glm_server.py
```

### Or using docker

```bash
docker compose up --build -d
```

The server will start on `http://localhost:8976` (or the port specified in your `.env` file).

### API Endpoints

#### V1 API (With Thinking Tags)

The v1 API returns responses that include thinking tags, showing the model's reasoning process:

```bash
# Chat completions with thinking tags
curl -X POST http://localhost:8976/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "0727-360B-API",
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "stream": false
  }'

# List available models
curl http://localhost:8976/v1/models
```

**Response format (v1):**
```json
{
  "choices": [{
    "message": {
      "content": "<think>Let me think about quantum computing...</think>\n\nQuantum computing is a revolutionary computing paradigm..."
    }
  }]
}
```

#### V2 API (Clean Responses)

The v2 API returns clean responses without thinking tags:

```bash
# Chat completions without thinking tags
curl -X POST http://localhost:8976/v2/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "0727-360B-API",
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "stream": false
  }'

# List available models
curl http://localhost:8976/v2/models
```

**Response format (v2):**
```json
{
  "choices": [{
    "message": {
      "content": "Quantum computing is a revolutionary computing paradigm..."
    }
  }]
}
```

### Using with OpenAI Python Client

```python
from openai import OpenAI

# Configure client to use your local GLM server
client = OpenAI(
    base_url="http://localhost:8976/v1",  # Use /v1 or /v2
    api_key="dummy-key"  # Any string works
)

# Use like normal OpenAI API
response = client.chat.completions.create(
    model="0727-360B-API",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message.content)
```

## Available Models

The server supports the following GLM models:

- `0727-360B-API` - Latest GLM-4 model with enhanced capabilities

## API Endpoints

### Chat Completions

**V1 Endpoint (with thinking tags):**
- `POST /v1/chat/completions`
- `GET /v1/models`

**V2 Endpoint (clean responses):**
- `POST /v2/chat/completions`
- `GET /v2/models`

### Additional Endpoints

- `GET /health` - Health check
- `POST /conversation/save` - Save conversation
- `POST /conversation/load` - Load conversation

## Response Format

### Non-Streaming Response

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "0727-360B-API",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}
```

### Streaming Response

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1677652288,
  "model": "0727-360B-API",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "Hello"
      },
      "finish_reason": null
    }
  ]
}
```

### V1 vs V2 Differences

**V1 API:** Returns responses with thinking tags when the model uses reasoning:
```
<think>Let me consider this question...</think>

The answer is...
```

**V2 API:** Returns clean responses with thinking content removed:
```
The answer is...
```

## Streaming Support

Both v1 and v2 APIs support streaming responses. Add `"stream": true` to your request:

```bash
curl -X POST http://localhost:8976/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "0727-360B-API",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

## Health Check

Check if the server is running:

```bash
curl http://localhost:8976/health
```

Response:
```json
{"status": "healthy", "timestamp": "2024-01-01T12:00:00Z"}
```

## Troubleshooting

### Common Issues

1. **Server won't start:**
   - Check if your `.env` file exists and contains a valid `GLM_API_TOKEN`
   - Ensure the port is not already in use
   - Check the console for error messages

2. **Invalid token error:**
   - Verify your GLM API token is correct
   - Make sure the token hasn't expired
   - Follow the token extraction steps in the Configuration section

3. **Connection refused:**
   - Ensure the server is running
   - Check if you're using the correct port
   - Verify firewall settings


## Requirements

- Python 3.10+
- requests >= 2.32.4
- flask >= 3.1.1
- python-dotenv >= 1.1.1

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source. Please check the license file for details.