from pydub import AudioSegment
import os
import openai
# import whisper

from faster_whisper import WhisperModel
model_size = "base.en"
# function to chunk and mp3 file into 24.5mb chunks
transcript = []


def split_audio(input_file, output_dir, chunk_size):
    audio = AudioSegment.from_file(input_file)
    total_duration = len(audio)
    
    # Calculate the chunk duration based on the audio's bit rate
    bit_rate = audio.frame_rate * audio.sample_width * audio.channels
    chunk_duration = (chunk_size * 5 * 1000) // bit_rate  # Chunk duration in milliseconds
    print(chunk_duration)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start = 0
    end = chunk_duration
    count = 1

    while start < total_duration:
        chunk = audio[start:end]
        output_file = os.path.join(output_dir, f"chunk_{count}.mp3")
        chunk.export(output_file, format="mp3", bitrate="64k")
        
        start += chunk_duration
        end += chunk_duration
        count += 1
        print("transcript",transcript)


# function to get transcribed text from the chunks
def transcribe(folder_path):
    print('transcribing...')
    for entry in os.listdir(folder_path):
        if entry.endswith(".mp3"):
            print(entry)
            audio_file = open(folder_path + "/" + entry, "rb")
            # result = openai.Audio.transcribe("whisper-1", audio_file)
            # model = whisper.load_model("base")
            model = WhisperModel(model_size, device="cuda", compute_type="float16")

            segments, info = model.transcribe("audio.mp3", beam_size=60)

            print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

            for segment in segments:
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                transcript.append(segment.text)
            # result = model.transcribe(folder_path + "/" + entry)
            # print(result["text"])
            
            
def convertMp3ToWav():
    # convert mp3 file to wav
    sound = AudioSegment.from_mp3("../audio-files/lex_ai_eliezer_yudkowsky.mp3")
    output = sound.export("output.wav", format="mp3", bitrate='64k')



if __name__ == "__main__":
    input_file = "../audio-files/lex_ai_eliezer_yudkowsky.mp3"
    output_file = "compressed_output.mp3"
    dir="chunks/"
    target_bitrate = "64k"  # You can adjust this value to your desired level of compression
    chunk_size = 20 * 1024 * 1024
    # split_audio(input_file, "chunks", chunk_size)
    transcribe(dir)
    # convertMp3ToWav()
    # f = open("eleiezer-lex.txt", "w")
    # result = ' '.join(transcript)
    # f.write(result)
    # f.close()
    