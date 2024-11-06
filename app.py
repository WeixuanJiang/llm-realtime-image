import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Query
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocketDisconnect
from openai import OpenAI
from dotenv import load_dotenv
import io
import soundfile as sf
import sounddevice as sd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PORT = int(os.getenv('PORT', 5050))
VOICE = os.getenv('VOICE')
OPENAI_MODEL = os.getenv('OPENAI_VOICE_MODEL')
OPENAI_BETA = os.getenv('OPENAI_BETA', "realtime=v1")
VISION_MODEL=os.getenv('OPENAI_VISION_MODEL')
app = FastAPI()

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

memory = []

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "AI Voice Assistant/Image Analyzer Server is running!"}

@app.get("/analyze-image", response_class=JSONResponse)
def analyze_image(image_path: str = Query(..., description="Path to the image to analyze")):
    """Analyze the content of an image using OpenAI."""
    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Encode the image
    base64_image = encode_image(image_path)

    # Create a request to OpenAI's API
    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is in this image?",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=300
        )

        # Extract the content from the response
        content = response.choices[0]  # This assumes the structure is `choices -> message -> content`
        print("Response content:", content)
        return {"response": content}
    except Exception as e:
        logging.error("Error parsing OpenAI response: %s", str(e))
        return {"error": str(e)}

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    logging.info("Client connected")
    await websocket.accept()

    async with websockets.connect(
        f"wss://api.openai.com/v1/realtime?model={OPENAI_MODEL}",
        extra_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": OPENAI_BETA
        }
    ) as openai_ws:
        logging.info("Connection with OpenAI established")
        await initialize_session(openai_ws)

        try:
            async def receive_and_forward_audio():
                """Receive audio from the client and send it to OpenAI."""
                while True:
                    audio_data = await websocket.receive_bytes()
                    audio_append = {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(audio_data).decode('utf-8')
                    }
                    await openai_ws.send(json.dumps(audio_append))

            async def receive_and_forward_response():
                """Receive audio response from OpenAI and handle it."""
                async for openai_message in openai_ws:
                    logging.info("Received message from OpenAI: %s", openai_message)
                    response = json.loads(openai_message)

                    # Check if the response contains the expected content
                    if 'response' in response and 'output' in response['response']:
                        output = response['response']['output']
                        if len(output) > 0 and 'content' in output[0] and len(output[0]['content']) > 0:
                            content = output[0]['content'][0]
                            if content['type'] == 'audio':
                                transcript = content.get('transcript', '')
                                logging.info("Transcript from OpenAI: %s", transcript)

                                # Append the assistant's response to memory
                                memory.append({"role": "assistant", "content": transcript})
                                
                                # Generate TTS and play it
                                generate_tts(transcript)
                        else:
                            logging.warning("Response does not contain expected content.")
                    else:
                        logging.warning("Response structure is not as expected: %s", response)

            # Run both coroutines concurrently
            await asyncio.gather(receive_and_forward_audio(), receive_and_forward_response())

        except WebSocketDisconnect:
            logging.info("Client disconnected.")

async def initialize_session(openai_ws):
    """Initialize the session with OpenAI, including conversation history."""
    # Prepare the conversation history for instructions
    history_text = "\n".join([f"{item['role']}: {item['content']}" for item in memory])
    instructions = f"""You are a helpful assistant. Here's the conversation so far:\n{history_text} 
                       Based on the above conversation, please respond thoughtfully and helpfully."""

    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "voice": VOICE,
            "instructions": instructions,
            "modalities": ["text", "audio"],
            "temperature": float(os.getenv('TEMPERATURE')),
        }
    }
    logging.info("Session initialized with OpenAI")
    logging.debug("Instructions sent to OpenAI: %s", instructions)
    await openai_ws.send(json.dumps(session_update))

def generate_tts(text):
    """Generate speech using OpenAI TTS and play the audio immediately."""
    try:
        # Generate the TTS response from OpenAI
        tts_response = client.audio.speech.create(
            model=os.getenv('VOICE_MODEL'),
            voice=VOICE,
            input=text,
            response_format=os.getenv('RESPONSE_FORMAT')
        )
        buffer = io.BytesIO()
        for chunk in tts_response.iter_bytes(chunk_size=4096):
            buffer.write(chunk)
        buffer.seek(0)

        with sf.SoundFile(buffer, 'r') as sound_file:
            data = sound_file.read(dtype='int16')
            sd.play(data, sound_file.samplerate)
            sd.wait()

    except Exception as e:
        logging.error("Error in generate_tts: %s", str(e))
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)