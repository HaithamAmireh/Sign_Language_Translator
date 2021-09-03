import speech_recognition as sr
def stt():
    recognizer = sr.Recognizer()
    how_long = 5

    with sr.Microphone() as source:
        print("Adjusting noise ")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print(f"Recording for {how_long} seconds")
        recorded_audio = recognizer.listen(source, timeout= how_long)
        print("Done recording")

    try:
        print("Recognizing the text")
        text = recognizer.recognize_google(
                recorded_audio, 
                language="en-US"
            )
        print(f"Decoded text: {text}")
        return text

    except Exception as ex:
        print(ex)
