import json
import os

CACHE_FILE = 'cache_data.json'

data = None


# Function to store data to the cache file
def store_to_cache(job_id, message_id, channel_id):
    global data
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as file:
            data = json.load(file)

    data[job_id] = {
        'message_id': message_id,
        'channel_id': channel_id
    }

    with open(CACHE_FILE, 'w') as file:
        json.dump(data, file)


def cache_job(job_id, message):
    store_to_cache(job_id, message.id, message.channel.id)


# Function to retrieve data from the cache file
def get_from_cache(job_id):
    global data

    if data is None:
        if not os.path.exists(CACHE_FILE):
            data = {}
        with open(CACHE_FILE, 'r') as file:
            data = json.load(file)

    return data.get(job_id, None)

def remove_from_cache(job_id):
    global data

    if data is None:
        if not os.path.exists(CACHE_FILE):
            data = {}
        with open(CACHE_FILE, 'r') as file:
            data = json.load(file)

    data.pop(job_id, None)

    with open(CACHE_FILE, 'w') as file:
        json.dump(data, file)


def get_cache():
    global data

    if data is None:
        if not os.path.exists(CACHE_FILE):
            data = {}
        with open(CACHE_FILE, 'r') as file:
            data = json.load(file)

    return data
