import streamlit as st
from moviepy.editor import VideoFileClip, AudioFileClip
from pydub import AudioSegment
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech_v1 as texttospeech
import openai
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Azure OpenAI API Keys and Endpoint
openai.api_type = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = "https://internshala.openai.azure.com/"
openai.api_version = "2024-08-01-preview"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_SAMPLE_RATE = 44100

# Define common filler words and non-verbal sounds
FILLER_WORDS = {"um", "uh", "like", "you know", "so", "well", "ah"}
NON_VERBAL_SOUNDS = {"cough", "scream", "breath", "sigh"}

def extract_and_compress_audio(video_path, output_format="wav", target_size_mb=10):
    """Extract audio from video and compress it."""
    temp_wav_path = os.path.join(CURRENT_DIR, "temp_audio.wav")
    compressed_audio_path = os.path.join(CURRENT_DIR, f"compressed_audio.{output_format}")

    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(temp_wav_path)

    audio = AudioSegment.from_wav(temp_wav_path)
    audio = audio.set_channels(1)

    duration_seconds = len(audio) / 1000.0
    target_bitrate = int((target_size_mb * 8 * 1024) / duration_seconds)

    audio.export(compressed_audio_path, format=output_format, bitrate=f"{target_bitrate}k")
    os.remove(temp_wav_path)

    return compressed_audio_path, video_clip.duration

def transcribe_audio(audio_path):
    """Transcribe the audio using Google Speech-to-Text with punctuation."""
    client = speech.SpeechClient()

    with open(audio_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=TARGET_SAMPLE_RATE,
        language_code="en-US",
        enable_automatic_punctuation=True,  # Enable punctuation during transcription
        enable_word_time_offsets=True
    )

    response = client.recognize(config=config, audio=audio)

    full_transcript = ""
    word_timings = []

    for result in response.results:
        full_transcript += result.alternatives[0].transcript + " "
        for word in result.alternatives[0].words:
            word_timings.append({
                'word': word.word,
                'start_time': word.start_time.total_seconds(),
                'end_time': word.end_time.total_seconds()
            })

    transcription_path = os.path.join(CURRENT_DIR, "transcription_with_punctuation.txt")
    with open(transcription_path, "w") as file:
        file.write(full_transcript)

    return full_transcript, word_timings

def remove_filler_words_and_add_pauses(text):
    """Remove filler words and replace them with pauses (full stops or ellipses)."""
    words = text.split()
    processed_words = []
    for word in words:
        clean_word = word.lower().strip(".,!?")
        if clean_word not in FILLER_WORDS and clean_word not in NON_VERBAL_SOUNDS:
            processed_words.append(word)
        elif clean_word in FILLER_WORDS:
            # Replace filler words with pauses to account for the time they consumed
            processed_words.append(".")  # Add a full stop (pause) where fillers were present
        elif clean_word in NON_VERBAL_SOUNDS:
            # Replace non-verbal sounds with ellipses for a longer pause
            processed_words.append("...")  # Add ellipsis to represent longer non-verbal sounds
    return " ".join(processed_words)

def correct_transcription(text, target_duration):
    """Send transcription to Azure OpenAI GPT-4 for grammar correction, keeping pauses."""
    # Remove filler words and replace them with pauses before sending to GPT-4
    text_with_pauses = remove_filler_words_and_add_pauses(text)

    # Instruct GPT-4 to maintain pauses and only improve grammar
    messages = [
        {"role": "system", "content": f"Correct this transcription while preserving pauses, punctuation, and the overall structure. Remove filler words and maintain pauses where filler words and non-verbal sounds existed."},
        {"role": "user", "content": f"Please correct the following text to fit within {target_duration:.2f} seconds while preserving the pauses:\n\n{text_with_pauses}"}
    ]

    response = openai.ChatCompletion.create(
        engine="gpt-4o",
        messages=messages,
        max_tokens=500
    )
    
    corrected_text = response.choices[0].message['content'].strip()

    corrected_path = os.path.join(CURRENT_DIR, "corrected_transcription_with_pauses.txt")
    with open(corrected_path, "w") as file:
        file.write(corrected_text)

    return corrected_text

def generate_audio_from_text(text, output_audio_path):
    """Generate audio from text using Google Text-to-Speech, leveraging pauses."""
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", name="en-US-Journey-D"
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
    response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

    with open(output_audio_path, "wb") as out:
        out.write(response.audio_content)

def replace_audio_in_video(video_path, audio_path, output_path):
    """Replace the audio in the video with the new synced audio."""
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    final_video = video_clip.set_audio(audio_clip)
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')

    video_clip.close()
    audio_clip.close()

def main():
    st.set_page_config(page_title="AI Audio Replacement", layout="wide")
    st.title("🎥 AI-Enhanced Video Audio Replacement App 🗣️")

    uploaded_file = st.file_uploader("Upload your video", type=["mp4", "mkv", "mov"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_file.write(uploaded_file.read())
            temp_video_path = temp_video_file.name

        try:
            audio_path, video_duration = extract_and_compress_audio(temp_video_path)
            st.write(f"Audio extracted and saved to: {audio_path}")
            st.audio(audio_path)

            # Transcribe original audio with punctuation
            with st.spinner("Transcribing original audio with punctuation..."):
                transcription, word_timings = transcribe_audio(audio_path)
            st.write(f"Transcription: {transcription}")

            # Correct transcription using GPT-4 while maintaining pauses for filler words
            corrected_text = correct_transcription(transcription, video_duration)
            st.write(f"Corrected Transcription: {corrected_text}")

            # Generate audio with pauses and punctuation
            generated_audio_path = os.path.join(CURRENT_DIR, "generated_audio_with_pauses.wav")
            generate_audio_from_text(corrected_text, generated_audio_path)
            st.audio(generated_audio_path)

            # Replace audio in the video
            output_video_path = os.path.join(CURRENT_DIR, "final_video.mp4")
            replace_audio_in_video(temp_video_path, generated_audio_path, output_video_path)

            st.success("Audio replaced successfully!")
            st.video(output_video_path)
            st.download_button(
                "Download Video", open(output_video_path, "rb").read(), file_name="final_video.mp4"
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.write("Please upload a video to get started.")

if __name__ == "__main__":
    main()
