import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Convert text to speech
text = "Jon went to winterfell and met with theon and they had a argument over the lord of the reach, this escalted and led to jon talking bad about his son"
engine.say(text)
engine.runAndWait()