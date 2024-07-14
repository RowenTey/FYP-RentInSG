from lib.utils.notify import send_message

def transform(**kwargs):
    # Your data transformation logic here
    send_message("Test", "This is a test message")
    print("Running data transformation script")