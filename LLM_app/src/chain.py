""" This module contains functions for loading a ConversationalRetrievalChain"""

import logging 

import wandb

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

from prompts import load_chat_prompt


logger = logging.getLogger(__name__)

def load_vector_store(wandb_run: wandb.run, openai_api_key: str) -> Chroma:
    """ 
    Load a vector store from a Weights & Biases artifact
    Args:
        run (wandb.run): An active Weights & Biases run
        openai_api_key (str): The OpenAI API key to use for embedding
    Returns:
        Chroma: A chroma vector store object
    """
    
    # Load vector artifacts
    
    vector_store_artifact_dir = wandb_run.use_artifact(
        wandb_run.config.vector_store_artifact, type="search_index"
    ).download()
    
    embedding_fn = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Load vector store
    vector_store = Chroma(
        embedding_function = embedding_fn, persist_directory=vector_store_artifact_dir 
    )
    return vector_store