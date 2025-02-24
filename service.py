import pika
import subprocess
import requests
import ast
import threading
import functools
from service_helper import send_progress
from service_helper import send_heartbeat
import os
import time
import random
import string

from pathlib import Path
Path("./documents").mkdir(parents=True, exist_ok=True)
Path("./converted").mkdir(parents=True, exist_ok=True)
Path("./results").mkdir(parents=True, exist_ok=True)

work_count = 0

worker_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
print("Worker ID:", end="")
print(worker_id)


def master_link():
    threading.Timer(5.0, master_link).start()
    send_heartbeat(worker_id)


master_link()


def run_model(connection, channel, delivery_tag, body):
    thread_id = threading.get_ident()

    # Get Payload Data
    payload = ast.literal_eval(body.decode('utf-8'))

    # print(payload)
    print("**********>> Queue %s starts <<**********" % payload['id'])

    # make payload.id a string
    payload['id'] = str(payload['id'])
    payload['project_id'] = str(payload['project_id'])
    if 'max_no_topic' in payload:
        payload['max_no_topic'] = str(payload['max_no_topic'])
    if 'criteria' in payload:
        payload['criteria'] = str(payload['criteria'])

    '''
    ==================================================================
    PART A : Preparing a Directory
    ==================================================================
    '''

    send_progress(
        id=payload['id'],
        code="010",
        keep=True)

    if not os.path.exists('./documents/' + payload['id']):
        os.mkdir("./documents/" + payload['id'])

    if not os.path.exists('./converted/' + payload['id']):
        os.mkdir("./converted/" + payload['id'])

    '''
    ==================================================================
    PART B : Setting Up A Thread
    ==================================================================
    '''
    send_progress(
        id=payload['id'],
        code="040",
        keep=True)

    print(" [D] Start Processing Queue-ID {} On Separate Thread".format(payload['id']))
    if 'criteria' not in payload:
        print(" [=] LDA")
        process = subprocess.Popen([
            'python',
            'main.py',
            payload['id'],
            payload['project_id'],
            payload['project_name'],
            payload['documents'],
            payload['max_no_topic']], stdout=subprocess.PIPE, cwd="/app")
    else:
        print(" [=] Similarity")
        process = subprocess.Popen([
            'python',
            'main.py',
            payload['id'],
            payload['project_id'],
            payload['project_name'],
            payload['documents'],
            payload['criteria']], stdout=subprocess.PIPE, cwd="/app/similarity")
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
        channel.basic_qos(prefetch_count=1)
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
