import os
import subprocess
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from dotenv import load_dotenv

# === LISTENING PARAMETERS ===
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
SILENCE_THRESHOLD = 30  # chunks of silence before stopping (~3 seconds)
MIN_SPEECH_CHUNKS = 2   # minimum chunks to consider as speech
# === === ===

load_dotenv()

PIPER_PATH = os.getenv("PIPER_PATH")
PIPER_VOICE = os.getenv("PIPER_VOICE")

# Load Whisper model
print("🎙️ Loading Whisper model...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
print("✅ Whisper ready!")


# ===== Speech to Text =====
def listen() -> str:
    """Record audio until the user stops speaking, then transcribe."""
    from silero_vad import load_silero_vad
    import torch

    vad_model = load_silero_vad()

    print("🎤 Listening...")

    audio_chunks = []
    silence_counter = 0
    speech_detected = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="float32", blocksize=CHUNK_SIZE) as stream:
        while True:
            chunk, _ = stream.read(CHUNK_SIZE)
            chunk_np = chunk.flatten()
            audio_chunks.append(chunk_np)

            # Check for speech in this chunk
            chunk_tensor = torch.tensor(chunk_np)
            speech_prob = vad_model(chunk_tensor, SAMPLE_RATE).item()

            if speech_prob > 0.5:
                speech_detected = True
                silence_counter = 0
            else:
                if speech_detected:
                    silence_counter += 1

            # Stop after enough silence following speech
            if speech_detected and silence_counter >= SILENCE_THRESHOLD:
                print("🔇 Speech ended, transcribing...")
                break

            # Safety cutoff at 30 seconds
            if len(audio_chunks) > (30 * SAMPLE_RATE / CHUNK_SIZE):
                print("⏱️ Max duration reached, transcribing...")
                break

    # Combine all chunks
    full_audio = np.concatenate(audio_chunks)

    # Save to temp file and transcribe
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, full_audio, SAMPLE_RATE)
        # No language parameter — Whisper auto-detects
        segments, _ = whisper_model.transcribe(tmp.name)
        transcript = " ".join([seg.text for seg in segments]).strip()

    os.unlink(tmp.name)
    print(f"📝 You said: {transcript}")
    return transcript


# ===== Text to Speech =====
def speak(text: str):
    """Convert text to speech using Piper and play it."""
    print(f"🔊 Speaking: {text[:50]}...")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        output_path = tmp.name

    # Call Piper executable
    subprocess.run(
        [PIPER_PATH, "--model", PIPER_VOICE, "--output_file", output_path],
        input=text.encode("utf-8"),
        check=True
    )

    # Play the audio
    data, sample_rate = sf.read(output_path)
    sd.play(data, sample_rate)
    sd.wait()

    os.unlink(output_path)


# ===== Voice Chat Loop =====
def voice_chat():
    """Full voice conversation loop."""
    from rag import ask

    speak("Hello! I am Chatette. How can I help you?")

    while True:
        try:
            # Listen for user input
            transcript = listen()

            if not transcript:
                speak("I didn't catch that. Could you repeat?")
                continue

            if any(word in transcript.lower() for word in ["goodbye", "bye", "exit", "quit"]):
                speak("Goodbye! Have a great day!")
                break

            # Get answer from RAG
            answer = ask(transcript)

            # Speak the answer
            speak(answer)

        except KeyboardInterrupt:
            print("\n🛑 Voice chat stopped.")
            break


if __name__ == "__main__":
    voice_chat()