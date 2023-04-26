import openai
import os
import tiktoken
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

tokenizer = tiktoken.get_encoding('cl100k_base')

def count_tokens(text):
    token_count = len(tokenizer.encode(text))
    return token_count


def chunk_text(text, max_token_size):
    tokens = text.split(" ")
    token_count = 0
    chunks = []
    current_chunk = ""

    for token in tokens:
        token_count += count_tokens(token)

        if token_count <= max_token_size:
            current_chunk += token + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = token + " "
            token_count = count_tokens(token)

    if current_chunk:
        chunks.append(current_chunk.strip())
    print("chunks", len(chunks))
    return chunks

def tk_len(text):
    token = tokenizer.encode (
        text,
        disallowed_special=()
    )
    return len(token)


def generate_summary(text):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that summarizes text."},
        {"role": "user", "content": f"Summarize the following text like a section of blog article while maintaining the context of the conversation: {text}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=200,  # Adjust based on your desired summary length
        n=1,
        stop=None,
        temperature=0.1,
    )

    summary = response.choices[0].message['content'].strip()
    print('summary-chunk')
    return summary

def refineSummary(text):
    prompt = "Summarize the following text in form of blog article divided into headers and subheaders while maintaining the context of the conversation:: \n \n " + text
    print('refining summary')


    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates medium like articles based on text provided in next messages while maintaining the context of the conversation. The format of output file should be a markdown file with headers as # and subheader as ##"},
            {"role": "user", "content": prompt},         
        ],
        n=1,
        stop=None,
        temperature=0.1,
    )
    refined_summary = completion.choices[0].message.content
    print('refined summary', refined_summary)
    return refined_summary


def summarize_large_text(input_text, output_file, max_token_size=4096):
    # Chunk the text into smaller parts
    text_chunks = chunk_text(input_text, max_token_size)
    print('chunks', len(text_chunks))
    
    # Generate summaries for each chunk
    summaries = [generate_summary(chunk) for chunk in text_chunks]
    print('summaries',len(summaries), output_file)
    # Combine the summaries into a single article
    article = "## Summary\n\n"
    for idx, summary in enumerate(summaries, 1):
        article += f"{summary}  \n\n"

    
    refinedSummary = refineSummary(article)
    print('refined summary', refinedSummary)
    # Save the article to a Markdown file
    with open(output_file, "w") as f:
        f.write(refinedSummary)


def summarize_large_text_langchain(input_text, output_file, max_token_size=4096):
    text_chunks = chunk_text(input_text, max_token_size)
    print('chunks', len(text_chunks))
    docs = [Document(page_content=t) for t in text_chunks]
    print('chunks', len(docs))
    prompt_template = """Summarize the following text like a section of medium app article: 


    {text}


    CONCISE SUMMARY:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    refine_template = (
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "Summarize the following text in form of medium app article divided into headers(##) and subheaders(#) and the output shoud be in .md format:: \n \n"
        "------------\n"
        "{text}\n"
        "------------\n"
        
    )
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )
    chain = load_summarize_chain(ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],temperature=0, model="gpt-3.5-turbo"),
        chain_type="refine", return_intermediate_steps=True, question_prompt=PROMPT, refine_prompt=refine_prompt)
    results = chain({"input_documents": docs}, return_only_outputs=True)
    print(results)

    with open(output_file, "w") as f:
        f.write(results['output_text'])