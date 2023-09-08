from telegram import Update
from telegram.ext import CommandHandler, MessageHandler, Filters, CallbackContext, Updater
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Загрузка модели и токенизатора
model = GPT2LMHeadModel.from_pretrained(r"C:\Users\genus\OneDrive\Desktop\Sirius_Tink\custom_ruDialoGPT")
tokenizer = GPT2Tokenizer.from_pretrained("tinkoff-ai/ruDialoGPT-medium")

def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Привет! Я ваш чат-бот. Задайте мне вопрос.')

def generate_response(user_input: str) -> str:
    input_ids = tokenizer.encode(user_input, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

def respond(update: Update, context: CallbackContext) -> None:
    user_input = update.message.text
    response = generate_response(user_input)
    update.message.reply_text(response)

def main() -> None:
    updater = Updater("6447422676:AAFNlOzZagIZiuuWw5n2vmpTa9WRY5EaGHM", use_context=True)

    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, respond))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()