import os
import time
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)


def is_downloadable(url):
    """
    Does the url contain a downloadable resource
    """
    try:
        h = requests.head(url, allow_redirects=True)
        header = h.headers
        content_type = header.get('content-type')
        if 'text' in content_type.lower():
            return False
        if 'html' in content_type.lower():
            return False
        return True
    except:
        return False


def send_progress(id, code, payload=None, keep=False, data=None, files=None):
    progress_payload = {'id': id, 'code': code, 'keep': keep}

    if "EXPRESS_HOST" in os.environ:
        api_url = "http://"+os.environ["EXPRESS_HOST"]+"/progress"
    else:
        api_url = "http://queueing-express:3000/progress"

    if payload is not None:
        progress_payload['payload'] = payload
    else:
        progress_payload['payload'] = ['']

    if data is not None:
        progress_payload['data'] = data

    sent = False
    while not sent:
        try:
            if files is None:
                r = session.post(api_url, progress_payload)
            else:
                r = session.post(api_url, json=progress_payload, files=files)
            sent = True
        except Exception as error:
            print(" [Worker->API] Trying to send ...")
            print(error)
            time.sleep(3)


def get_status_message(status_code):
    statuses = {
        '010': "preparing request's directory",
        '011': "preparing document{}",
        '020': "downloading request's resources",
        '021': "{} is being downloaded",
        '022': "{} is being converted",
        '030': "converting request's resources",
        '040': "threading",
        '050': "complete exporting file in {} language",

        '110': "input dataset",
        '111': "input dataset from {}",

        '120': "cleaning data and creating word tokenize",
        '121': "cleaning document {}",
        '122': "splitting word in document {}",

        '130': "creating bag of words",
        '140': "modeling",
        '150': "cleaning environment",
        '160': "evaluating model",
        '170': 'exporting to html format',
        '180': 'converting exported html to Thai',

        '410': "{} is not downloadable",

        '510': "input dataset failed on document {}"
    }

    return statuses.get(str(status_code), "Unknown Error")
