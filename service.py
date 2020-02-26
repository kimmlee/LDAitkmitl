import pika
import subprocess
import requests
import ast
import threading
import functools
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from service_helper import send_progress
import os
import time

from pathlib import Path
Path("./documents").mkdir(parents=True, exist_ok=True)
Path("./converted").mkdir(parents=True, exist_ok=True)
Path("./results").mkdir(parents=True, exist_ok=True)

def run_model(connection, channel, delivery_tag, body):
    thread_id = threading.get_ident()

    # Get Payload Data
    payload = ast.literal_eval(body.decode('utf-8'))
    print(payload)
    print("**********>> Queue %s starts <<**********" % payload['id'])

    # make payload.id a string
    payload['id'] = str(payload['id'])
    payload['max_no_topic'] = str(payload['max_no_topic'])

    # Prepare session to make requests
    session = requests.Session()

    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)

    '''
    ==================================================================
    PART A : Setting Up A Thread
    ==================================================================
    '''
    send_progress(
        session=session,
        id=payload['id'],
        code="040",
        keep=True)

    print(" [D] Start Processing Queue-ID {} On Separate Thread".format(payload['id']))
    process = subprocess.Popen(['python', 'main.py', payload['id'], payload['documents'], payload['max_no_topic']], stdout=subprocess.PIPE, cwd="/app")
    for line in process.stdout:
        print(line.decode('utf-8').replace('\n', ''))

    '''
    ==================================================================
    PART E : Finishing Request
    ==================================================================
    '''
    print("")
    print("***********>> Queue %s ends <<***********" % payload['id'])
    print("")

    cb = functools.partial(ack_message, channel, delivery_tag)
    connection.add_callback_threadsafe(cb)


def ack_message(channel, delivery_tag):
    """Note that `channel` must be the same pika channel instance via which
    the message being ACKed was retrieved (AMQP protocol constraint).
    """
    if channel.is_open:
        channel.basic_ack(delivery_tag)
    else:
        # Channel is already closed, so we can't ACK this message;
        # log and/or do something that makes sense for your app in this case.
        pass


def callback(channel, method, properties, body, args):
    (connection, threads) = args
    delivery_tag = method.delivery_tag
    t = threading.Thread(target=run_model, args=(connection, channel, delivery_tag, body))
    t.start()
    threads.append(t)


threads = []

'''
##################
# RabbitMQ Setup #
##################
'''

username = 'ml-rabbitmq'
password = 'passwordpala'

is_connected = False
while True:
    try:
        credentials = pika.PlainCredentials(username, password)
        connection = pika.BlockingConnection(pika.ConnectionParameters(
            os.getenv("RABBITMQ_HOST", 'queueing-rabbitmq'), 
            int(os.getenv("RABBITMQ_PORT", '5672')), 
            '/', 
            credentials))

        callback_hook = functools.partial(callback, args=(connection, threads))
        channel = connection.channel()
        channel.basic_qos(prefetch_count=3)
        channel.basic_consume(queue="processing.requests", on_message_callback=callback_hook)
        channel.start_consuming()

        if not is_connected:
            print(' [*] Waiting for messages.')
            is_connected = True
    # Don't recover if connection was closed by broker
    except pika.exceptions.ConnectionClosedByBroker:
        break
    # Don't recover on channel errors
    except pika.exceptions.AMQPChannelError:
        break
    # Recover on all other connection errors
    except pika.exceptions.AMQPConnectionError:
        print(" [E] Trying to connect... ")
        time.sleep(3)
        is_connected = False
        continue

for thread in threads:
    thread.join()
