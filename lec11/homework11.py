import speech_recognition as sr

def transcribe_wavefile(filename, language):
    '''
    Use sr.Recognizer.AudioFile(filename) as the source,
    recognize from that source,
    and return the recognized text using Google's engine.
    
    @params:
    filename (str) - the filename from which to read the audio
    language (str) - the language of the audio (e.g., 'en-US', 'zh-TW')
    
    @returns:
    text (str) - the recognized speech, or an empty string if recognition fails.
    '''
    r = sr.Recognizer()
    
    try:
        # Open the audio file using sr.AudioFile
        with sr.AudioFile(filename) as source:
            # Record the entire audio data from the file
            audio = r.record(source) 
            
        # Use Google Web Speech API for recognition, specifying the language
        text = r.recognize_google(audio, language=language)
        
        return text
        
    except sr.UnknownValueError:
        # Google Speech Recognition could not understand audio
        return "" 
    except sr.RequestError as e:
        # Could not request results from Google Speech Recognition service
        print(f"Google Speech Recognition API Error: {e}")
        return ""
    except Exception as e:
        # Handle other potential errors (e.g., file not found, incompatible format)
        print(f"An unexpected error occurred: {e}")
        return ""