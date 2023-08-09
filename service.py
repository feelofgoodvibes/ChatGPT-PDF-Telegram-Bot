import os
from typing import Optional
from settings import Settings

from telegram import Update
from telegram.ext import ContextTypes

from pypdf import PdfReader
from pypdf.errors import PdfReadError


def verify_pdf(path: str) -> Optional[PdfReader]:
    """Verify that file is valid PDF document
    
    Returns instance of `PdfReader` if document is valid, else `None`
    """

    try:
        return PdfReader(path)
    except PdfReadError:
        return None


async def save_document_from_update(update: Update, context: ContextTypes.DEFAULT_TYPE) -> tuple[str, str, str]:
    document = update.message.document
    document_name = document.file_name

    destination_directory = os.path.join(".",
                                    Settings.USER_FILES_DIRECTORY.value,
                                    str(update.message.from_user.id))
    
    destination_path = os.path.join(destination_directory, document_name)

    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    bot_file = await context.bot.get_file(document)
    await bot_file.download_to_drive(custom_path=destination_path)

    return (document_name, destination_directory, destination_path)
