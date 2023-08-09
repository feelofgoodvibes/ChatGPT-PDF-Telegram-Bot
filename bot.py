from settings import Settings
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import service
import langchain_llm

# Dict of users LLM chains
# When user sends PDF document, program will create LLM chain based on that document
# and that LLM chain will be assigned to that user.
# Later, this LLM chain will be retrieved from dict and used to generate response
# specifically to that particular user
USERS_LLM_CHAINS = dict()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""

    user = update.effective_user
    await update.message.reply_html(f"Hey, {user.full_name} 👋\n\n📄 Send me any PDF file you want me to analyze!\n\n❗Keep in mind, I'm working only with documents up to 50 pages")


async def pdf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Accept PDF document from the user"""

    # Saving document to disk
    document_name, _, destination_path = await service.save_document_from_update(update, context)

    # Verifying PDF document
    pdf = service.verify_pdf(destination_path)
    
    # If PDF is not valid - terminate process
    if pdf is None:
        await update.message.reply_text(f"⚠️ This file is not a valid PDF document. I can't work with it 😞")
        return
    
    # If PDF has more than 50 pages - terminate process
    if len(pdf.pages) > 50:
        await update.message.reply_text(f"⚠️ This document has more than 50 pages.  I can't work with it 😞")
        return

    await update.message.reply_text(f"👍 Document is valid (Pages: {len(pdf.pages)})\n\nNow give me some time to read it. I'll notify you when I'm done 😉")

    # Create LLM chain for current user
    user_chain = langchain_llm.create_chain(update.message.from_user.id, destination_path)

    # Saving LLM chain to dict of users chains to use it later on every user prompt
    USERS_LLM_CHAINS[update.message.from_user.id] = user_chain

    await update.message.reply_text(f"🎉 I'm ready to start talking with you about document {document_name}\n\n🔎 You can ask your questions now")


async def wrong_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("😞 I'm working only with PDF documents")


async def prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Listen for user prompts after recieving PDF document"""

    user_prompt = update.message.text

    # Get current user LLM chain
    user_chain = USERS_LLM_CHAINS.get(update.message.from_user.id)

    # If bot hasn't created chain for user,
    # it will tell the user to send PDF document first
    if user_chain is None:
        await update.message.reply_text(f"🤔 I don't know what you are talking about yet. Send me the PDF document you are want to talk about.")
        return

    # Create inital message that will be edited as LLM generates response
    init_message = await update.message.reply_text("🕑")

    # Call LLM Chain with user prompt
    await user_chain.acall(user_prompt,
                           callbacks=[langchain_llm.AnswerGenerationCallback(init_message)],
                           return_only_outputs=True)


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(Settings.BOT_TOKEN.value).build()

    application.add_handler(CommandHandler("start", start))

    # Handler to accept PDF document from the user
    application.add_handler(MessageHandler(filters.Document.PDF & ~filters.COMMAND, pdf))
    
    # Handler to accept prompt from the user
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, prompt))

    # Handler to accept any attachment except PDF document to tell user that bot working only with PDF documents
    application.add_handler(MessageHandler(filters.ATTACHMENT & ~filters.Document.PDF, wrong_file))

    # Run the bot until the user presses Ctrl-C
    print("Bot is listening")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
