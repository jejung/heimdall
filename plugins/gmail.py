import httplib2
import os

from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage


SCOPES = 'https://www.googleapis.com/auth/gmail.readonly'
CLIENT_SECRET_FILE = 'data/client_secret.json'
APPLICATION_NAME = 'Heimdall'


def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir, 'heimdall.json')
    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        credentials = tools.run_flow(flow, store)
    return credentials


def get_emails(max_samples=0):
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    service = discovery.build('gmail', 'v1', http=http)

    currently_samples = 1
    results = service.users().messages().list(userId='me').execute()
    while 'nextPageToken' in results:
        for row in results['messages']:
            message_data = service.users().messages().get(userId='me', id=row['id']).execute()
            label_data = None
            for label_id in message_data['labelIds']:
                if 'CATEGORY_' in label_id:
                    label_data = label_id
                    break
            if label_data is not None:
                currently_samples += 1
                email_message = message_data['snippet'].encode('utf-8')
                data = {
                    'label': label_data,
                    'body': email_message
                }
                yield data
                if 0 < max_samples < currently_samples:
                    raise StopIteration()
        results = service.users().messages().list(userId='me', pageToken=results['nextPageToken']).execute()
