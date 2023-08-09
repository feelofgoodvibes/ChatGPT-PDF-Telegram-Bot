import os
from settings import Settings

from langchain import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import Chroma
from langchain.callbacks.base import BaseCallbackHandler
from telegram import Message


class AnswerGenerationCallback(BaseCallbackHandler):
    """Custom CallbackHandler.
    
    This handler accepts `init_message` that will be edited
    as LLM generates text in streaming mode"""

    def __init__(self, init_message: Message):
        self.init_message = init_message

        # Amount of generated tokens
        self.iteration = 0
    
        # Full text of the LLM output
        self.answer = ""

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when an LLM in streaming mode produces a new token
        
        New token is appened to current text of the message
        """

        self.iteration += 1
        self.answer += token

        # Edit telegram message every 4th token
        if self.iteration % 4 == 0:
            await self.init_message.edit_text(self.answer)

    async def on_llm_end(self, response, **kwargs) -> None:
        """Insert full text to message at the end of an LLM run"""

        await self.init_message.edit_text(self.answer)


def create_chain(user_id: int, document_path: str) -> RetrievalQAWithSourcesChain:
    """Create chain for user with `user_id` based on document"""
    
    document_loader = PyPDFLoader(document_path)
    document_pages = document_loader.load_and_split(splitter)
    
    chroma_presist_directory = os.path.join(Settings.USER_FILES_DIRECTORY.value, str(user_id), "db")
    chromadb = Chroma.from_documents(document_pages, embeddings, persist_directory=chroma_presist_directory)
    chromadb.persist()

    qa = RetrievalQAWithSourcesChain.from_chain_type(llm=openai,
                                                     chain_type='stuff',
                                                     retriever=chromadb.as_retriever(),
                                                     reduce_k_below_max_tokens=True,
                                                     max_tokens_limit=4096,
                                                     chain_type_kwargs={"prompt": prompt})

    return qa


openai = ChatOpenAI(openai_api_key=Settings.OPENAI_KEY.value, model=Settings.MODEL_NAME.value, streaming=True)
embeddings = OpenAIEmbeddings(openai_api_key=Settings.OPENAI_KEY.value, model=Settings.EMBEDDING_NAME.value)
splitter = CharacterTextSplitter(chunk_size=2048, chunk_overlap=0)

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{summaries}

Question: {question}
Answer: 
"""

prompt = PromptTemplate(template=template, input_variables=["summaries", "question"])
