import time
import logging
import os
import collections
import csv
import queue

import deepspeech
import numpy as np
import pyaudio
import wave
import webrtcvad
from halo import Halo
from scipy import signal
from TTS.api import TTS  
import sounddevice as sd  

logging.basicConfig(level=20)

class Audio:
    """Streams raw audio from the microphone."""
    
    FORMAT = pyaudio.paInt16
    RATE_PROCESS = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, device=None, input_rate=RATE_PROCESS, file=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            if self.chunk is not None:
                in_data = self.wf.readframes(self.chunk)
            callback(in_data)
            return (None, pyaudio.paContinue)

        if callback is None:
            callback = lambda in_data: self.buffer_queue.put(in_data)

        self.buffer_queue = queue.Queue()
        self.device = device
        self.input_rate = input_rate
        self.sample_rate = self.RATE_PROCESS
        self.block_size = int(self.RATE_PROCESS / float(self.BLOCKS_PER_SECOND))
        self.block_size_input = int(self.input_rate / float(self.BLOCKS_PER_SECOND))
        self.pa = pyaudio.PyAudio()

        kwargs = {
            'format': self.FORMAT,
            'channels': self.CHANNELS,
            'rate': self.input_rate,
            'input': True,
            'frames_per_buffer': self.block_size_input,
            'stream_callback': proxy_callback,
        }

        self.chunk = None
        if self.device:
            kwargs['input_device_index'] = self.device
        elif file is not None:
            self.chunk = 320
            self.wf = wave.open(file, 'rb')

        self.stream = self.pa.open(**kwargs)
        self.stream.start_stream()

    def resample(self, data, input_rate):
        """Resample from input_rate to RATE_PROCESS."""
        data16 = np.frombuffer(data, dtype=np.int16)
        resample_size = int(len(data16) / self.input_rate * self.RATE_PROCESS)
        resample = signal.resample(data16, resample_size)
        resample16 = np.array(resample, dtype=np.int16)
        return resample16.tobytes()

    def read_resampled(self):
        """Return a block of audio data resampled to 16000hz."""
        return self.resample(data=self.buffer_queue.get(), input_rate=self.input_rate)

    def read(self):
        """Return a block of audio data."""
        return self.buffer_queue.get()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=3, device=None, input_rate=None, file=None):
        super().__init__(device=device, input_rate=input_rate, file=file)
        self.vad = webrtcvad.Vad(aggressiveness)

    def frame_generator(self):
        """Yield all audio frames from the microphone."""
        if self.input_rate == self.RATE_PROCESS:
            while True:
                yield self.read()
        else:
            while True:
                yield self.read_resampled()

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Yield series of audio frames comprising each utterance."""
        if frames is None:
            frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            if len(frame) < 640:
                return

            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()
            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()

# TTS Setup using Glow-TTS
tts_engine = TTS(model_name="tts_models/en/ljspeech/glow-tts")

def play_audio(audio_data, sample_rate=22050):
    """Play the generated audio."""
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()

def speak_text(text, audio_stream):
    """Speak the given text and mute the microphone during playback."""
    mute_microphone(audio_stream)  
    audio = tts_engine.tts(text)  
    play_audio(audio)  
    unmute_microphone(audio_stream)  

def mute_microphone(audio_stream):
    """Mute the microphone by stopping the audio stream."""
    audio_stream.stream.stop_stream()

def unmute_microphone(audio_stream):
    """Unmute the microphone by starting the audio stream."""
    audio_stream.stream.start_stream()

def save_to_csv(codice_fiscale, reference_number):
    """Save Codice Fiscale and Reference Number into a CSV file."""
    filename = 'client_data.csv'
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as file:
        fieldnames = ['codice_fiscale', 'reference_number']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # Write header if the file doesn't exist

        writer.writerow({'codice_fiscale': codice_fiscale, 'reference_number': reference_number})
    logging.info(f"Data saved to {filename}")

def recognize_speech(model, vad_audio):
    """Recognize speech using DeepSpeech and VAD."""
    frames = vad_audio.vad_collector()
    stream_context = model.createStream()
    wav_data = bytearray()

    spinner = Halo(spinner='line')
    for frame in frames:
        if frame is not None:
            spinner.start()
            logging.debug("streaming frame")
            stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
            wav_data.extend(frame)
        else:
            spinner.stop()
            logging.debug("end utterance")
            text = stream_context.finishStream()
            return text

def main(ARGS):
    # Load DeepSpeech model
    logging.info("Loading DeepSpeech model...")
    model = deepspeech.Model(ARGS.model)
    if ARGS.scorer:
        logging.info("Using external scorer: %s", ARGS.scorer)
        model.enableExternalScorer(ARGS.scorer)

    # Start audio with VAD
    vad_audio = VADAudio(aggressiveness=ARGS.vad_aggressiveness,
                         device=ARGS.device,
                         input_rate=ARGS.rate,
                         file=ARGS.file)

    # Asking for Codice Fiscale
    speak_text("Please tell me your Codice Fiscale.", vad_audio)
    time.sleep(1)  
    logging.info("Listening for Codice Fiscale...")
    codice_fiscale = recognize_speech(model, vad_audio)
    print(f"Recognized Codice Fiscale: {codice_fiscale}")

    # Confirming Codice Fiscale
    speak_text(f"You provided: {codice_fiscale}. Now, please provide your reference number.", vad_audio)
    time.sleep(1)  
    logging.info("Listening for Reference Number...")
    reference_number = recognize_speech(model, vad_audio)
    print(f"Recognized Reference Number: {reference_number}")

    # Saving both pieces of data into a CSV file
    save_to_csv(codice_fiscale, reference_number)

if __name__ == '__main__':
    DEFAULT_SAMPLE_RATE = 16000

    import argparse
    parser = argparse.ArgumentParser(description="Stream from microphone to DeepSpeech using VAD")

    parser.add_argument('-v', '--vad_aggressiveness', type=int, default=3,
                        help="Set aggressiveness of VAD: an integer between 0 and 3.")
    parser.add_argument('--nospinner', action='store_true',
                        help="Disable spinner")
    parser.add_argument('-w', '--savewav',
                        help="Save .wav files of utterances to given directory")
    parser.add_argument('-f', '--file',
                        help="Read from .wav file instead of microphone")

    parser.add_argument('-m', '--model', required=True,
                        help="Path to the model (protocol buffer binary file, or entire directory containing all standard-named files for model)")
    parser.add_argument('-s', '--scorer',
                        help="Path to the external scorer file.")
    parser.add_argument('-d', '--device', type=int, default=None,
                        help="Device input index.")
    parser.add_argument('-r', '--rate', type=int, default=DEFAULT_SAMPLE_RATE,
                        help=f"Input device sample rate. Default: {DEFAULT_SAMPLE_RATE}.")

    ARGS = parser.parse_args()
    if ARGS.nospinner:
        logging.disable(logging.INFO)

    main(ARGS)
