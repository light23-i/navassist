import speech_recognition as sr
from os import path
import shutil
from pydub import AudioSegment
import pyttsx3
from deepface import DeepFace
import whisper
import sys
import os
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv

def record():
# Sampling frequency
    freq = 44100
    
    # Recording duration
    duration = 4
    
    # Start recorder with the given values
    # of duration and sample frequency
    recording = sd.rec(int(duration * freq),
                    samplerate=freq, channels=2)
    
    # Record audio for the given number of seconds
    sd.wait()
    
    # This will convert the NumPy array to an audio
    # file with the given sampling frequency
    write("recording0.wav", freq, recording)
    
    # Convert the NumPy array to audio file
    wv.write("recording1.wav", recording, freq, sampwidth=2)
# objs = DeepFace.analyze(img_path = "t.png", 
#         actions = ['age', 'gender', 'race', 'emotion']
# )
# initialisation
model = whisper.load_model("base")
def say(text):
    engine = pyttsx3.init()
    
    # testing
    engine.say(text)
    #engine.say("Thank you, Geeksforgeeks")
    engine.runAndWait()
# convert mp3 file to wav
p = "f.mp3"                            
audio = whisper.load_audio(p)
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio).to(model.device)
options = whisper.DecodingOptions(fp16=False)
a = whisper.decode(model, mel, options)
x = a.text

#a = text("p.mp3")

y = x.split(" ")
# if "face" in a:
#      g = objs[0]['dominant_gender']
#      t = f"gender is {g}"
#      say(t)
if "detect" in y:
    dfs = DeepFace.find(img_path = "p1.png",enforce_detection =False, db_path = "ga")
    print(dfs)
    if len(dfs[0])!=0:
        p = dfs[0]['identity'][0].split('.')
        s = "Name is " + p[0]
        say(s)
        print(s)
    else:
        y = len(os.listdir('ga'))
        say('Do you want to save the information about the person. Say YES OR NO')
        record()
        audio = whisper.load_audio("p1.mp3")
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions(fp16=False)
        a = whisper.decode(model, mel, options)
        g = a.text
        pp = g.split(" ")
        print(pp)
        if "repeat" in pp:
            record()
            audio = whisper.load_audio("recording1.wav")
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            options = whisper.DecodingOptions(fp16=False)
            a = whisper.decode(model, mel, options)
            g = a.text
            pp = g.split(" ")
        if "yes." or "Yes," or "yes" or "Yes" in pp:
            say('Tell Name of the person')
            record()
            audio = whisper.load_audio("recording1.wav")
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            options = whisper.DecodingOptions(fp16=False)
            a = whisper.decode(model, mel, options)
            g = a.text
            print(g)
            os.rename('p1.png',f'{g} {y+1}.png')
            shutil.copy(f'{g} {y+1}.png','ga')
            say('person added to database')
            os.remove('ga/representations_vgg_face.pkl')
    
