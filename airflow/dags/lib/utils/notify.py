def send_message(service_name: str, message: str):
    import os
    import logging
    import telegram
    
    TOKEN = os.environ["TELEGRAM_TOKEN"]
    CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

    bot = telegram.Bot(token=TOKEN)
    bot.send_message(chat_id=CHAT_ID, text=f"FYP-{service_name} - {message}")
    logging.info("Message sent!")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    send_message("Test", "This is a test message")
