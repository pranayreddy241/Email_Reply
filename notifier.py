import os
from twilio.rest import Client
ACCOUNT_SID=os.getenv('TWILIO_ACCOUNT_SID'); AUTH_TOKEN=os.getenv('TWILIO_AUTH_TOKEN'); FROM=os.getenv('TWILIO_FROM'); TO=os.getenv('OWNER_PHONE')
def notify_owner(message:str):
    if not all([ACCOUNT_SID,AUTH_TOKEN,FROM,TO]):
        raise RuntimeError('Missing Twilio env vars')
    Client(ACCOUNT_SID,AUTH_TOKEN).messages.create(from_=FROM,to=TO,body=message)
