"""A simple chatbot that uses LangChain and Gradio UI to answer questions about wandb documentation."""


import os
from types import SimpleNamespace

import gradio as gr
import wandb
from chain import get_answer, load_chain, load_vector_store
from config import default_config 


