import requests
import os
import time

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


def send_progress(session, id, code, payload=None, keep=False, files=None):
    progress_payload = {'id': id, 'code': code, 'keep': keep}

    if "EXPRESS_HOST" in os.environ:
        api_url = "http://"+os.environ["EXPRESS_HOST"]+"/progress"
    else:
        api_url = "http://queueing-express:3000/progress"

    if payload is not None:
        progress_payload['payload'] = payload
    else:
        progress_payload['payload'] = ['']

    
    sent = False
    while not sent:
        try:
            if files is None:
                r = session.post(api_url, progress_payload)
            else:
                r = session.post(api_url, progress_payload, files=files)
            sent = True
        except:
            print(" [Worker->API] Trying to send ...")
            time.sleep(3)




def get_status_message(status_code):
    statuses = {
        '010': "preparing request's directory",
        '011': "preparing document{}",
        '020': "downloading request's resources",
        '021': "{} is being downloaded",
        '030': "converting request's resources",
        '040': "threading",
        '050': "end thread",

        '110': "input dataset",
        '111': "input dataset from {}",

        '120': "cleaning data and creating word tokenize",
        '121': "cleaning document {}",
        '122': "splitting word in document {}",

        '130': "creating bag of words",
        '140': "modeling",
        '150': "cleaning environment",

        '410': "{} is not downloadable",

        '510': "input dataset failed on document {}"
    }

    return statuses.get(str(status_code), "Unknown Error")
