from enum import Enum
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

class Settings(Enum):
    # OPEN AI key from https://platform.openai.com/account/api-keys
    OPENAI_KEY = os.environ.get("OPENAI_KEY")

    # Model name from https://platform.openai.com/docs/models/overview
    MODEL_NAME = "gpt-3.5-turbo"
    EMBEDDING_NAME = "text-embedding-ada-002"

    BOT_TOKEN = os.environ.get("BOT_TOKEN")

    # Name of the directory where all the user files (documents and Chroma databases) will be stored
    USER_FILES_DIRECTORY = "USER_FILES"