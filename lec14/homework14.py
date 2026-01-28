import gtts
import speech_recognition as sr
import librosa
import soundfile as sf
import os

def synthesize(text, lang, filename):
    '''
    Use gtts.gTTS(text=text, lang=lang) to synthesize speech, then write it to filename.
    
    @params:
    text (str) - the text you want to synthesize
    lang (str) - the language in which you want to synthesize it
    filename (str) - the filename in which it should be saved
    '''
    # Initialize the Google Text-to-Speech object
    tts = gtts.gTTS(text=text, lang=lang)
    
    # Save the synthesized audio as an MP3 file
    tts.save(filename)

def make_a_corpus(texts, languages, filenames):
    '''
    Create many speech files, and check their content using SpeechRecognition.
    The output files should be created as MP3, then converted to WAV, then recognized.

    @param:
    texts - a list of the texts you want to synthesize
    languages - a list of their languages
    filenames - a list of their root filenames, without the ".mp3" ending

    @return:
    recognized_texts - list of the strings that were recognized from each file
    '''
    recognized_texts = []
    recognizer = sr.Recognizer()

    for text, lang, root_filename in zip(texts, languages, filenames):
        mp3_file = root_filename + ".mp3"
        wav_file = root_filename + ".wav"

        # 1. Synthesize text to MP3 format
        synthesize(text, lang, mp3_file)

        # 2. Convert MP3 to WAV format for better compatibility with SpeechRecognition
        # librosa.load reads the audio, and sf.write saves it as a WAV file
        y, sr_native = librosa.load(mp3_file, sr=None)
        sf.write(wav_file, y, sr_native)

        # 3. Perform Speech Recognition on the generated WAV file
        with sr.AudioFile(wav_file) as source:
            audio_data = recognizer.record(source)
            try:
                # Use Google Speech Recognition API to convert audio back to text
                result = recognizer.recognize_google(audio_data, language=lang)
                recognized_texts.append(result)
            except (sr.UnknownValueError, sr.RequestError):
                # Append an empty string if recognition fails or a request error occurs
                recognized_texts.append("")

    return recognized_texts