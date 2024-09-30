# import os
# from groq import Groq
# import re
# import requests
# import torch

# import warnings
# warnings.filterwarnings("ignore")

# import numpy as np
# import matplotlib.pyplot as plt

# from chromadb.api.types import EmbeddingFunction
# from dotenv import load_dotenv

# from ibm_watson_machine_learning.foundation_models import Model
# from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

# from langchain.document_loaders import PyPDFLoader
# from sentence_transformers import SentenceTransformer
# from sklearn.manifold import TSNE
# from sklearn.neighbors import NearestNeighbors
# from typing import Literal, Optional, Any

# # %%
# def pdf_to_text(path: str, 
#                 start_page: int = 1, 
#                 end_page: Optional[int | None] = None) -> list[str]:
#     """
#     Converts PDF to plain text.

#     Params:
#         path (str): Path to the PDF file.
#         start_page (int): Page to start getting text from.
#         end_page (int): Last page to get text from.
#     """
#     loader = PyPDFLoader("NHPS-POLICIES-HR.pdf")
#     pages = loader.load()
#     total_pages = len(pages)

#     if end_page is None:
#         end_page = len(pages)

#     text_list = []
#     for i in range(start_page-1, end_page):
#         text = pages[i].page_content
#         text = text.replace('\n', ' ')
#         text = re.sub(r'\s+', ' ', text)
#         text_list.append(text)

#     return text_list

# # %%
# # PDF files available:
# #    "pdfs/pie_recipe.pdf"
# #    "pdfs/paper_flowers.pdf"

# text_list = pdf_to_text("NHPS-POLICIES-HR.pdf")

# def text_to_chunks(texts: list[str], 
#                    word_length: int = 300, 
#                    start_page: int = 1) -> list[list[str]]:
#     """
#     Splits the text into equally distributed chunks.

#     Args:
#         texts (str): List of texts to be converted into chunks.
#         word_length (int): Maximum number of words in each chunk.
#         start_page (int): Starting page number for the chunks.
#     """
#     text_toks = [t.split(' ') for t in texts]
#     chunks = []

#     for idx, words in enumerate(text_toks):
#         for i in range(0, len(words), word_length):
#             chunk = words[i:i+word_length]
#             if (i+word_length) > len(words) and (len(chunk) < word_length) and (
#                 len(text_toks) != (idx+1)):
#                 text_toks[idx+1] = chunk + text_toks[idx+1]
#                 continue
#             chunk = ' '.join(chunk).strip() 
#             chunk = f'[Page no. {idx+start_page}]' + ' ' + '"' + chunk + '"'
#             chunks.append(chunk)
            
#     return chunks


# chunks = text_to_chunks(text_list)

# # %%
# #%%time
# # Load the model from TF Hub
# class MiniLML6V2EmbeddingFunction(EmbeddingFunction):
#     MODEL = SentenceTransformer('all-MiniLM-L6-v2')
#     def __call__(self, texts):
#         return MiniLML6V2EmbeddingFunction.MODEL.encode(texts).tolist()
# emb_function = MiniLML6V2EmbeddingFunction()

# # %%
# def get_text_embedding(texts: list[list[str]], 
#                        batch: int = 1000) -> list[Any]:
#         """
#         Get the embeddings from the text.

#         Args:
#             texts (list(str)): List of chucks of text.
#             batch (int): Batch size.
#         """
#         embeddings = []
#         for i in range(0, len(texts), batch):
#             text_batch = texts[i:(i+batch)]
#             # Embeddings model
#             emb_batch = emb_function(text_batch)
#             embeddings.append(emb_batch)
#         embeddings = np.vstack(embeddings)
#         return embeddings

# # %%
# embeddings = get_text_embedding(chunks)


# # %%

# # # Create a t-SNE model
# # tsne = TSNE(n_components=2, random_state=42)
# # embeddings_with_question = np.vstack([embeddings, emb_question])
# # embeddings_2d = tsne.fit_transform(embeddings_with_question)

# # %%
# #embeddings_2d.shape

# # %%
# def visualize_embeddings(embeddings_2d: np.ndarray, 
#                          question: Optional[bool] = False, 
#                          neighbors: Optional[np.ndarray] = None) -> None:
#     """
#     Visualize 384-dimensional embeddings in 2D using t-SNE, label each data point with its index,
#     and optionally plot a question data point as a red dot with the label 'q'.

#     Args:
#         embeddings (numpy.array): An array of shape (num_samples, 384) containing the embeddings.
#         question (numpy.array, optional): An additional 384-dimensional embedding for the question.
#                                           Default is None.
#     """

#     # Scatter plot the 2D embeddings and label each data point with its index
#     plt.figure(figsize=(10, 8))
#     num_samples = embeddings.shape[0]
#     if neighbors is not None:
#         for i, (x, y) in enumerate(embeddings_2d[:num_samples]):
#             if i in neighbors:
#                 plt.scatter(x, y, color='purple', alpha=0.7)
#                 plt.annotate(str(i), xy=(x, y), xytext=(5, 2), textcoords='offset points', color='black')
#             else:
#                 plt.scatter(x, y, color='blue', alpha=0.7)
#                 plt.annotate(str(i), xy=(x, y), xytext=(5, 2), textcoords='offset points', color='black')
#     else:
#         for i, (x, y) in enumerate(embeddings_2d[:num_samples]):
#             plt.scatter(x, y, color='blue', alpha=0.7)
#             plt.annotate(str(i), xy=(x, y), xytext=(5, 2), textcoords='offset points', color='black')
        
#     # Plot the question data point if provided
#     if question:
#         x, y = embeddings_2d[-1]  # Last point corresponds to the question
#         plt.scatter(x, y, color='red', label='q')
#         plt.annotate('q', xy=(x, y), xytext=(5, 2), textcoords='offset points', color='black')

#     plt.title('t-SNE Visualization of 384-dimensional Embeddings')
#     plt.xlabel('Dimension 1')
#     plt.ylabel('Dimension 2')
#     plt.show()

# # # %%
# # #visualize_embeddings(embeddings_2d[:-1])

# # # %%
# # #visualize_embeddings(embeddings_2d, True)

# # # %%
# # nn_2d = NearestNeighbors(n_neighbors=5)
# # nn_2d.fit(embeddings_2d[:-1])

# # # %%
# # neighbors = nn_2d.kneighbors(embeddings_2d[-1].reshape(1, -1), return_distance=False)

# # # %%
# # #visualize_embeddings(embeddings_2d, True, neighbors)

# # # %%
# # nn = NearestNeighbors(n_neighbors=5)
# # nn.fit(embeddings)



# # # %%
# # neighbors = nn.kneighbors(emb_question, return_distance=False)
# # #neighbors

# # # %%
# # topn_chunks = [chunks[i] for i in neighbors.tolist()[0]]

# # %%
# def build_prompt(question,topn_chunks):
#     prompt = ""
#     prompt += 'Search results:\n'
    
#     for c in topn_chunks:
#         prompt += c + '\n\n'
    
#     prompt += "Instructions: You are a conversational AI assistant that is provided a list of documents and a user query to answer based on information from the documents. The user also provides an answer mode which can be 'Grounded' or 'Mixed'. For answer mode Grounded only respond with exact facts from documents, for answer mode Mixed answer using facts from documents and your own knowledge. Cite all facts from the documents using <co: doc_id></co> tags."\
#             "Compose a comprehensive reply to the query using the search results given. "\
#             "Cite each reference using [Page Number] notation (every result has this number at the beginning). "\
#             "Citation should be done at the end of each sentence. If the search results mention multiple subjects "\
#             "with the same name, create separate answers for each. Only include information found in the results and "\
#             "don't add any additional information. Make sure the answer is correct and don't output false content. "\
#             "If the text does not relate to the query, simply state 'Found Nothing'. Ignore outlier "\
#             "search results which has nothing to do with the question. Only answer what is asked. The "\
#             "answer should be short and concise." 
    
#     prompt += f"\n\n\nQuery: {question}\n\nAnswer: "
    
#     return prompt

# def send_to_llm2(input):
    
#     client = Groq(
#     api_key="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"
#     )

#     chat_completion = client.chat.completions.create(
#     messages=[
#         {
#             "role": "user",
#             "content": input,
#         }
#     ],
#     model="llama-3.1-70b-versatile",
#     )

#     return chat_completion.choices[0].message.content

# # %%
# def custom_query(question):

#     emb_function = MiniLML6V2EmbeddingFunction()
#     emb_question = emb_function([question])
#     nn = NearestNeighbors(n_neighbors=5)
#     nn.fit(embeddings)
#     neighbors = nn.kneighbors(emb_question, return_distance=False)
#     topn_chunks = [chunks[i] for i in neighbors.tolist()[0]]
#     prompt = build_prompt(question,topn_chunks)
#     answer=send_to_llm2(prompt)
#     return answer

import os
import re
import requests
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt
from groq import Groq
from dotenv import load_dotenv
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from typing import Optional, Any

warnings.filterwarnings("ignore")

# %% Define function to convert PDF to text
def pdf_to_text(path: str, 
                start_page: int = 1, 
                end_page: Optional[int | None] = None) -> list[str]:
    """
    Converts PDF to plain text.
    Params:
        path (str): Path to the PDF file.
        start_page (int): Page to start getting text from.
        end_page (int): Last page to get text from.
    """
    loader = PyPDFLoader(path)
    pages = loader.load()
    total_pages = len(pages)

    if end_page is None:
        end_page = total_pages

    text_list = []
    for i in range(start_page - 1, end_page):
        text = pages[i].page_content
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)
        text_list.append(text)

    return text_list

# %% Load text from PDF
text_list = pdf_to_text("NHPS-POLICIES-HR.pdf")

# %% Define function to split text into chunks
def text_to_chunks(texts: list[str], 
                   word_length: int = 300, 
                   start_page: int = 1) -> list[list[str]]:
    """
    Splits the text into equally distributed chunks.
    Args:
        texts (str): List of texts to be converted into chunks.
        word_length (int): Maximum number of words in each chunk.
        start_page (int): Starting page number for the chunks.
    """
    text_toks = [t.split(' ') for t in texts]
    chunks = []

    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i + word_length]
            if (i + word_length) > len(words) and (len(chunk) < word_length) and (
                len(text_toks) != (idx + 1)):
                text_toks[idx + 1] = chunk + text_toks[idx + 1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[Page no. {idx + start_page}] ' + '"' + chunk + '"'
            chunks.append(chunk)

    return chunks

# %% Create chunks from text
chunks = text_to_chunks(text_list)

# %% Load the model for embeddings
class EmbeddingFunction:
    MODEL = SentenceTransformer('all-MiniLM-L6-v2')

    @classmethod
    def get_embeddings(cls, texts):
        return cls.MODEL.encode(texts).tolist()

# %% Get text embeddings
def get_text_embedding(texts: list[list[str]], 
                       batch: int = 1000) -> list[Any]:
    """
    Get the embeddings from the text.
    Args:
        texts (list[str]): List of chunks of text.
        batch (int): Batch size.
    """
    embeddings = []
    for i in range(0, len(texts), batch):
        text_batch = texts[i:(i + batch)]
        emb_batch = EmbeddingFunction.get_embeddings(text_batch)
        embeddings.append(emb_batch)
    embeddings = np.vstack(embeddings)
    return embeddings

# %% Get embeddings for chunks
embeddings = get_text_embedding(chunks)

# %% Define function to build the prompt for LLM
def build_prompt(question, topn_chunks):
    prompt = "Search results:\n"
    for c in topn_chunks:
        prompt += c + '\n\n'
    
    prompt += ("Instructions: You are a conversational AI assistant that is provided a list of documents and "
               "a user query to answer based on information from the documents. The user also provides an "
               "answer mode which can be 'Grounded' or 'Mixed'. For answer mode Grounded only respond with "
               "exact facts from documents, for answer mode Mixed answer using facts from documents and your "
               "own knowledge. Cite all facts from the documents using <co: doc_id></co> tags. Compose a "
               "comprehensive reply to the query using the search results given. Cite each reference using "
               "[Page Number] notation (every result has this number at the beginning). Citation should be done "
               "at the end of each sentence. If the search results mention multiple subjects with the same name, "
               "create separate answers for each. Only include information found in the results and don't add any "
               "additional information. Make sure the answer is correct and don't output false content. If the "
               "text does not relate to the query, simply state 'Found Nothing'. Ignore outlier search results "
               "which has nothing to do with the question. Only answer what is asked. The answer should be short "
               "and concise.")
    
    prompt += f"\n\n\nQuery: {question}\n\nAnswer: "
    
    return prompt

# %% Define function to send the prompt to the LLM
def send_to_llm2(input):
    client = Groq(
        api_key="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"
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": input,
            }
        ],
        model="llama-3.1-70b-versatile",
    )

    return chat_completion.choices[0].message.content

# %% Define custom query function
def custom_query(question):
    emb_function = EmbeddingFunction()
    emb_question = emb_function.get_embeddings([question])
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(embeddings)
    neighbors = nn.kneighbors(emb_question, return_distance=False)
    topn_chunks = [chunks[i] for i in neighbors.tolist()[0]]
    prompt = build_prompt(question, topn_chunks)
    answer = send_to_llm2(prompt)
    return answer
