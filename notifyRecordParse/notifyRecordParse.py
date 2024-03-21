import pyaudio
import wave
import whisper
import sys

class NotifyRecordParse:
    CHUNK = 1024

    def initialize(self):
        FILENAME = "notifyRecordParse\prompt.wav"
        self.wf = wave.open(FILENAME, 'rb')
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format = self.p.get_format_from_width(self.wf.getsampwidth()),
            channels = self.wf.getnchannels(),
            rate = self.wf.getframerate(),
            output = True
        )

    def activate(self):
        FILENAME = "prompt.wav"
        REPLY = "reply.wav"
        self.initialize()
        self.play()
        self.stop()
        self.startRecording(REPLY)
        response = self.parseRecording(REPLY)
        return self.isCallNeeded(response)

    def play(self):
        """ Play entire file """
        data = self.wf.readframes(self.CHUNK)
        while data != b'':
            self.stream.write(data)
            data = self.wf.readframes(self.CHUNK)

    def stop(self):
        """ Graceful shutdown """ 
        self.stream.close()
        self.p.terminate()

    def startRecording(self, filePath):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        frames = []
        seconds = 5
        CHUNK = 1024
        p = pyaudio.PyAudio()
        stream = p.open(format = FORMAT
                , channels = CHANNELS
                , rate = RATE
                , input = True
                , frames_per_buffer = CHUNK)
        print("start recording")
        for i in range(0, int (RATE / CHUNK * seconds)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("stop recording")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(filePath, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    
    def parseRecording(self, filePath):
        model = whisper.load_model("tiny")
        transcription = model.transcribe(filePath)
        return transcription["text"]
    
    def isCallNeeded(self, response):
        response_lowercase = response.lower()
        print(response_lowercase)
        if "yes" in response_lowercase or not response_lowercase:
            return True
        else:
            return False

# a = NotifyRecordParse()
# print(a.activate())
