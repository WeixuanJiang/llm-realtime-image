# Real-Time AI Voice Assistant & Image Analyzer

A Python-based application that combines real-time voice interaction and image analysis capabilities using OpenAI's advanced AI models.

## Features

- **Real-time Voice Communication**
  - WebSocket-based voice streaming
  - Text-to-Speech (TTS) conversion
  - Conversation memory for contextual responses

- **Image Analysis**
  - Real-time image content analysis
  - Support for various image formats
  - Detailed image description generation

## Prerequisites

- Python 3.8+
- OpenAI API Key
- Required Python packages:
  - fastapi
  - websockets
  - openai
  - python-dotenv
  - uvicorn
  - pydantic

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in a `.env` file:
```env
OPENAI_API_KEY=your_api_key_here
PORT=5050
VOICE=your_preferred_voice
OPENAI_VOICE_MODEL=your_voice_model
OPENAI_VISION_MODEL=your_vision_model
```

## Usage

1. Start the server:
```bash
python main.py
```

2. The server will start on the specified port (default: 5050)

3. Available endpoints:
   - `/`: Health check endpoint
   - `/analyze-image`: POST endpoint for image analysis
   - `/media-stream`: WebSocket endpoint for real-time voice communication

## API Documentation

### Image Analysis
Send a GET request to `/analyze-image` with the image path as a query parameter.

### Voice Communication
Connect to the WebSocket endpoint at `/media-stream` for real-time voice interaction.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
