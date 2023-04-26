import os
import io
import whisper
from flask import Flask, request, render_template
from pytube import YouTube
from pydub import AudioSegment
import subprocess
import textwrap
import requests
import openai
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain


from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import VectorDBQA
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.llms import GPT4All

from faster_whisper import WhisperModel
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import vectordb
# import pandas as pd
# import numpy as np
# from openai.embeddings_utils import get_embedding, cosine_similarity
import json
import tiktoken
from hyperdb import HyperDB
import hashlib
import summarisePodcast



m = hashlib.md5()
tokenizer = tiktoken.get_encoding('cl100k_base')

model = whisper.load_model('base.en')  # seleting the base
text_splitter = CharacterTextSplitter().from_tiktoken_encoder(
    chunk_size=1500, chunk_overlap=20)

model_size = "base.en"
# Run on GPU with FP16
faster_model = WhisperModel(model_size, compute_type="int8",  cpu_threads=16)

chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    # Optional, defaults to .chromadb/ in the current directory
    persist_directory="../chromadb",
))


app = Flask(__name__)


def trim_audio_chunks(input_file, output_dir, max_chunk_size_mb=24):
    # Extract audio from the file path
    audio = AudioSegment.from_file(input_file)
    total_duration = len(audio)

    # Calculate the chunk duration based on the audio's bit rate
    bit_rate = audio.frame_rate * audio.sample_width * audio.channels
    # Chunk duration in milliseconds (45 minutes)
    chunk_duration = (15 * 60 * 1000)
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


@app.route("/")
def search_form():
    """Function to render the search form"""
    return render_template("search_form.html")


@app.route("/get-podcast-audio", methods=['GET'])
async def get_podcast_audio():
    """Function to summarise transcript"""
    file_url = request.args.get("file")
    summary = ''
    with open(file_url, 'r') as file:
        contents = file.read()
        summary = await summarisePodcast.summarize_large_text(contents, 'chunks/new_summary.md')
    return summary


@app.route("/summarise", methods=['GET'])
async def summarise():
    """Function to summarise transcript"""
    file_url = request.args.get("file")
    summary = ''
    with open(file_url, 'r') as file:
        contents = file.read()
        summary = summarisePodcast.summarize_large_text(contents, 'transcription-workspace/new_summary_fast.md')
    return summary


@app.route("/transcribe")
async def transcribe():
    """Function to get audio file and transcribe it"""
    # get url from firestore and download audio

    podcast_episode_url = request.args.get("url")
    audio_file = requests.get(podcast_episode_url)

    if audio_file.status_code == 200:
        # Save the downloaded file
        with open('../episode.mp3', "wb") as file:
            file.write(audio_file.content)
    print('file downloaded')

    # # Splitting the audio into chunks and saving in chunks folder
    # trim_audio_chunks('./episode.mp3', './transcription-workspace')
    # print('chunks saved')
    # transcripts = []
    # segments, info = faster_model.transcribe('episode.mp3', beam_size=3)
    # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    # for segment in segments:
    #     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        # transcript.append(segment.text)
    
    # print('transcription done', " ".join(transcripts))
    # ## reading chunked files from chunks folder and transcribing them
    # audio_chunks = []
    # for entry in os.listdir('./transcription-workspace'):
    #     if entry.endswith(".mp3"):
    #         audio_chunks.append('./transcription-workspace/' + entry)

    # print(audio_chunks)
    # results = model.transcribe('episode.mp3')
    # with open("transcription-workspace/transcript.txt", 'w', encoding='utf-8') as file:
    #         file.write(results['text'])

    # print("entry 1",results[0]['text'])
    # print("entry 2",results[1]['text'])
    # print("entry 3",results[2]['text'])
    # print("entry 4",results[3]['text'])
    # print("entry 5",results[4]['text'])
    # print("entry 6",results[5]['text'])
    # print("entry 7",results[6]['text'])
    # print("entry 8",results[7]['text'])
    # print("entry 9",results[8]['text'])
    # print('transcription complete')

    # function to concat transcript dict texts
    # transcritpion_text = ''
    # for item in transcription:
    #     text += item
    #     with open("transcription-workspace/transcript.txt", 'w', encoding='utf-8') as file:
    #         file.write(text)

    # with open("transcription-workspace/transcript.txt", 'w', encoding='utf-8') as file:
    #         file.write(transcription['text'])

    # with open('transcription-workspace/transcript.txt', 'r') as file:
    #          contents = file.read()
    #          save_transcript_chunks_vector(contents)

    # local_db = vectordb.saveFiles(
    #     filenames=["transcription-workspace/transcript.txt"])
    # print(local_db)
    # query_response = vectordb.summarise()
    # print(query_response)
    # remove chunk files after transcription
    # for entry in os.listdir('./chunks'):
    #     if entry.endswith(".mp3") or entry.endswith(".txt"):
    #         os.remove("./transcription-workspace/" + entry)
    # print('transcription', transcription)

    # return transcription


# def summarize(text):
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": "Summarize the following text:\n\n{text}"}],
#         max_tokens=2500,
#         n=1,
#         stop=None,
#         temperature=0.5,
#     )
#     return response.choices[0].message.content

# def chunk_text(text, max_length=2500):
#     return textwrap.wrap(text, max_length)

# def summarize_large_text(text, chunk_size=2500):
#     chunks = chunk_text(text, chunk_size)
#     summaries = [summarize(chunk) for chunk in chunks]
#     print(len(summaries))
#     text= " ".join(summaries)
#     refineSummary(text)

# def create_chat_completion(messages, temperature=0.5, max_tokens=None)->str:
#     """Create a chat completion using the OpenAI API"""
#     response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=messages,
#             temperature=temperature
#         )

#     return response.choices[0].message["content"]


# changes for chunking data and saving to chroma db

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


def transcript_chunks(documents, max_length=1000):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,
        chunk_overlap=20
    )
    return text_splitter.split_text(documents)


def save_transcript_chunks_vector(text, max_length=1000):
    chunks = transcript_chunks(text, max_length)
    ids = []
    metadatas = []
    for i, chunk in enumerate(chunks):
        uid = m.hexdigest()[:12]
        ids.append(f'{uid}-{i}')
        metadatas.append({'source': 'huberman lab podcast'})
    chroma_vector_store(chunks, ids, metadatas)
    # with open('hyperDB/pod_episode.jsonl', 'w') as f:
    #     for doc in documents:
    #         f.write(json.dumps(doc) + '\n')
    # # print(documents)
    # db = initialise_local_db('hyperDB/pod_episode.jsonl')
    # db.save("hyperDB/podcasts_hyperdb.json")


def chroma_vector_store(chunks, ids, metadatas):
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ['OPENAI_API_KEY'],
        model_name="text-embedding-ada-002"
    )
    # embeddings = OpenAIEmbeddings()
    podcast_collection = chroma_client.create_collection(
        name="podcasts", embedding_function=openai_ef)
    podcast_collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids,
    )
    coll_ref = chroma_client.get_collection(
        name='podcasts', embedding_function=openai_ef)
    print(coll_ref)
    # vectordb = Chroma.from_documents(texts, embeddings)
    # llm_model = OpenAI(
    #         model="text-davinci-003",
    #         temperature=0.2
    #     )
    # qa = VectorDBQA.from_chain_type(llm=llm_model, chain_type="stuff", vectorstore=vectordb)
    # print('vector db created')
    # query = "what is this text about? give me a summary in 500 words"
    # result = qa.run(query)
    # print(result)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
