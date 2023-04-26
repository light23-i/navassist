import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 3  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
print(sd.query_devices())
sd.wait()  # Wait until recording is finished
print(myrecording)
write('output.wav', fs, myrecording)