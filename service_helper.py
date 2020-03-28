import os
import time
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import string
import datetime

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


def send_heartbeat(worker_id):

    if "EXPRESS_HOST" in os.environ:
        api_url = "http://"+os.environ["EXPRESS_HOST"]+":3000/api/health/worker"
    else:
        api_url = "http://queueing-express:3000/health/worker"
    try:
        session.post(api_url, json={"worker_id": worker_id})
    except Exception as error:
        time.sleep(5)


def filename_from_request(project_id):
    now = datetime.datetime.now()
    name = [str(project_id).zfill(9), now.strftime('%Y%m%d'), now.strftime('%H%M')]
    name = "_".join(name)

    filenames = {
        "th": name + "_th.html",
        "en": name + "_en.html"
    }

    return filenames


def send_progress(id, code, payload=None, keep=False, data=None, files=None):
    progress_payload = {'id': id, 'code': code, 'keep': keep}
    print(" |-[Send Progress] => ", end="")
    print(progress_payload)

    if "EXPRESS_HOST" in os.environ:
        api_url = "http://"+os.environ["EXPRESS_HOST"]+":3000/api/progress"
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
                r = session.post(api_url, json=progress_payload)
            else:
                r = session.post(api_url, progress_payload, files=files)
            sent = True
        except Exception as error:
            print(" |-[Bridge] Trying to send ...")
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
        '050': "complete",

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

        "190": "pushing result to API",
        "191": "pushing files",
        "192": "pushing result",

        '410': "{} is not downloadable",

        '510': "input dataset failed on document {}",
        "601": "document download process is failed due to unreachable url(s)",
        "602": "duplicated urls were found on the payload"
    }

    return statuses.get(str(status_code), "Unknown Error")
