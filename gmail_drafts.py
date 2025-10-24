import base64, os, pickle
from email.message import EmailMessage
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
SCOPES=['https://www.googleapis.com/auth/gmail.compose']
def _svc():
    creds=None
    if os.path.exists('token.pickle'):
        with open('token.pickle','rb') as f: creds=pickle.load(f)
    if not creds or not getattr(creds,'valid',False):
        if creds and getattr(creds,'refresh_token',None): creds.refresh(Request())
        else:
            flow=InstalledAppFlow.from_client_secrets_file('credentials.json',SCOPES)
            creds=flow.run_local_server(port=0)
        with open('token.pickle','wb') as f: pickle.dump(creds,f)
    return build('gmail','v1',credentials=creds)

def create_draft(from_addr, subject, body, in_reply_to=None, to_addr=None):
    s=_svc(); m=EmailMessage(); m['From']=from_addr
    if to_addr: m['To']=to_addr
    m['Subject']=subject
    if in_reply_to: m['In-Reply-To']=in_reply_to; m['References']=in_reply_to
    m.set_content(body)
    raw=base64.urlsafe_b64encode(m.as_bytes()).decode()
    s.users().drafts().create(userId='me', body={'message': {'raw': raw}}).execute()
