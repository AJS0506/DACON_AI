import os, glob
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from FlagEmbedding import BGEM3FlagModel
from langchain.schema import Document
import pickle, pymupdf4llm

import pandas as pd
import tabula

from dotenv import load_dotenv
load_dotenv()

import transformers
import torch, random

from tqdm import tqdm


# =====================================
CHUNK_SIZE = 500
OVER_SIZE = 100
EMBED_BATCH = 50
# =====================================




model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

def get_embed(titles, batch_k = EMBED_BATCH):
    embeddings = []
    original_titles = []

    for idx in range(0, len(titles), batch_k):
        start = idx
        end = min(idx + batch_k, len(titles))

        embedding = model.encode(titles[start:end],
                                    batch_size=batch_k,
                                    max_length=8192,
                                    # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                    )['dense_vecs']
        original_titles.extend(titles[start:end])
        embeddings.extend(embedding)

    return original_titles, embeddings

def get_table(pdf_path : str) -> list[str]:
    table_str = ""

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=OVER_SIZE, 
        length_function=len,
        is_separator_regex=False,
    )

    dfs = tabula.read_pdf(
        pdf_path, 
        pages="all", 
        stream=True,
        lattice = False,
        guess=True,
        multiple_tables=True,
    )

    for index, table in enumerate(dfs):
        table_str += (str(table)) + "\n\n"

    # 테이블을 청크로 나눔
    #table_data = text_splitter.create_documents([table_str])
    #table_data = [doc.page_content for doc in table_data]

    # 하나의 테이블을 한개의 청크로 함.
    table_data = table_str.split("\n\n")
    return table_data

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

def get_markdown(pdf_path : str) -> list[str]:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVER_SIZE,
        length_function=len,
        is_separator_regex=False,
        #separators=["##", "\n\n", "\n", " ", ""],
    )

    markdown = pymupdf4llm.to_markdown(pdf_path)    
    texts = text_splitter.create_documents([markdown])
    
    markdown_strs = [text.page_content for text in texts]

    return markdown_strs 

from pdfminer.high_level import extract_text

def get_miner(pdf_path : str) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=OVER_SIZE, 
        length_function=len,
        is_separator_regex=False,
    )
        
    text = extract_text(pdf_path)

    texts = text_splitter.create_documents([text])
    texts_str = [text.page_content for text in texts]
    
    return texts_str


paths = ["train_source","test_source"]

for path in paths:

    pdf_files = glob.glob("{}/*.pdf".format(path))

    for pdf_path in pdf_files:

        table_datas = get_table(pdf_path)
        markdown_datas = get_markdown(pdf_path)
        #miner_datas = get_miner(pdf_path)
        
        print(pdf_path)
        print(len(table_datas))
        print(len(markdown_datas))

        table_sentence, table_embed = get_embed(table_datas)
        markdown_sentence, markdown_embed = get_embed(markdown_datas)

        table_pairs = []
        for s,e in zip(table_sentence,table_embed):
            table_pairs.append([s,e])

        markdown_pairs = []
        for s,e in zip(markdown_sentence, markdown_embed):
            markdown_pairs.append([s,e])

        merged_pairs = table_pairs + markdown_pairs

        with open("embed_500_100/{}_table.pkl".format(pdf_path.split("/")[1]),'wb') as file:
            pickle.dump(table_pairs, file)

        with open("embed_500_100/{}_markdown.pkl".format(pdf_path.split("/")[1]),'wb') as file:
            pickle.dump(markdown_pairs, file)

        with open("embed_500_100/{}_merged.pkl".format(pdf_path.split("/")[1]),'wb') as file:
            pickle.dump(merged_pairs, file)
