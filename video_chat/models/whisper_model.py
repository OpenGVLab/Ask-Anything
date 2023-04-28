import os
import whisper

def has_intersection(t1, t2):
    if t1[1] < t2[0] or t2[1] < t1[0]:
        return False
    else:
        return True

class AudioTranslator():
    def __init__(self, model='base', device='cuda'):
        self.device = device
        self.model = whisper.load_model(model).to(device)

    def __call__(self, video_path):
        print("Extract the audio results.")
        audio_results = self.model.transcribe(video_path)["segments"]
        print("Finished.")
        return audio_results
    
    def match(self, audio_results, start, end):
        transcript = ''
        for res in audio_results:
            if has_intersection((start, end), (res["start"], res["end"])):
                transcript += res['text'] + ' '
        return transcript
