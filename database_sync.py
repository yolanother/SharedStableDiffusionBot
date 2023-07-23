import re
import traceback

from google.cloud.firestore_v1 import ArrayUnion

import getimageinfo
from sdbot_config_manager import dsref, dbref
from art_data import Art, Avatar, Author, Prompt, Parameters
from webapi import post

doc = dsref.collection(u'art').document(u'artwork')

def sync_midjourney_message(message):
    author = Author.from_discord_message(message)
    arts = Art.from_midjourney_message(message)
    for art in arts:
        data = art.to_dict()
        post("submit", data)
        #dsref.collection(u'art').document(art.id).set(data)
        #dsref.collection(u'prompts').document(art.parameters.prompt.prompt.replace('/', '')).set({u'images': ArrayUnion([art.to_ref()])}, merge=True)
        #dsref.collection(u'models').document(art.model).set({u'images': ArrayUnion([art.to_ref()])}, merge=True)
        (size, (x, y)) = getimageinfo.getsizes(data['url'])
        #dsref.collection(u'art').document(art.id).set({u'width': x, u'height': y}, merge=True)
        index_art_words(art)

    #dsref.collection(u'authors').document(author.id).set(author.to_dict())

def sync_job_by_name(jobName, markSynced=True):
    job = dbref.child("jobs").child("data").child(jobName).get()
    if job is not None and 'job' in job and 'job-synced' not in job['job']:
        job['name'] = jobName
        sync_job(job)
        if markSynced:
            dbref.child('jobs').child('data').child(jobName).child('job').child('job-synced').set(True)
            dbref.child('jobs').child('queue').child(jobName).delete()


def sync_job(job):
    print (f'Syncing {job["name"]}...')
    author = Author.from_job(job)
    try:
        arts = Art.from_job(job)
        if arts is not None:
            images = []
            for art in arts:
                data = art.to_dict()
                print("Syncing to artapi...")
                post("submit", data, job['name'])
                print("done.")

            print ("Syncing to firestore...")
            for art in arts:
                images.append(art.to_ref())
                #try:
                #    dsref.collection(u'art').document(art.id).set(data)
                #    dsref.collection(u'models').document(art.model).set({u'images': ArrayUnion([art.to_ref()])}, merge=True)
                #except Exception as e:
                #    print(f"Error: {e}")
            print("done.")

            prompt = Prompt.from_job(job)
            #if len(images) > 0:
            #    dsref.collection(u'prompts').document(prompt.prompt).set({u'images': ArrayUnion(images)}, merge=True)

            #if author is not None and author.id is not None:
            #    dsref.collection(u'authors').document(f"{author.id}").set(author.to_dict())
    except Exception as e:
        print (e)
        traceback.print_exc()


def sync_realtime():
    ignored = True
    records = dbref.child("records").child("all").get()
    for name in records.keys():
        if not ignored or name == '1009605988443234334':
            ignored = False
            record = records[name]
            if 'author-id' in record:
                author = Author.from_values(record['author-id'], record['username'], record['mention'], record['avatar'])
            else:
                author = None
            if record['model'] == 'midjourney':
                parameters = Parameters.from_string(record['prompt'])
            elif 'parameters' in record:
                parameters = Parameters.from_dict(record['parameters'])
            elif 'prompt' in record:
                parameters = Parameters.from_string(record['prompt'])
            else:
                parameters = Parameters("", {})

            parameters.parameters['upscaled'] = record['upscaled']
            art = Art(name, record['url'], author, parameters, model=record['model'],
                      width=record['width'] if 'width' in record else None,
                      height=record['height'] if 'height' in record else None)

            print (f'Syncing {art.id}...')
            #print (art.to_dict())
            #dsref.collection(u'art').document(f'{art.id}').set(art.to_dict())
            #dsref.collection(u'prompts').document(art.parameters.prompt.prompt.replace('/', '')).set({u'images': ArrayUnion([art.to_ref()])}, merge=True)

def sync_jobs():
    jobs = dbref.child("jobs").child("data").get()
    for name in jobs.keys():
        job = jobs[name]
        job['name'] = name
        if 'job' in job and 'status' in job['job'] and job['job']['status'] == 'complete':
            sync_job(job)

def flush_queue():
    jobs = dbref.child("jobs").child("queue").get()
    for name in jobs.keys():
        job = jobs[name]
        dbref.child("jobs").child("queue").child(job['name']).delete()
        dbref.child("jobs").child("completed").child(job['name']).set(job)

def fix_timestamps():
    print('Fixing timestamps...')
    records = dsref.collection(u'art').where('width', '==', -1).stream()
    print(f"Records: {records}")
    for r in records:
        record = r.to_dict()
        print(r.id)
        print(r.to_dict())
        (size, (x, y)) = getimageinfo.getsizes(record['url'])
        print(f"size: {x}x{y}")
        dsref.collection(u'art').document(r.id).set({u'width': x, u'height': y}, merge=True)

def sanatize_prompt(prompt):
    prompt = prompt.lower()
    prompt = re.sub(r'http\S+', '', prompt)
    prompt = re.sub(r'@\S+', '', prompt)
    prompt = re.sub(r'[^\w]', ' ', prompt)
    prompt = re.sub(r'\s+', ' ', prompt)
    return prompt.strip()

def index_record_words(record, words):
    prompt = record['parameters']['prompt']
    for word in sanatize_prompt(prompt).split(' '):
        word = word.strip()
        if len(word) > 2:
            if word not in words:
                words[word] = []
            words[word].append({
                'id': record['id'],
                'url': record['url']
            })

def index_art_words(art):
    words = {}
    record = art.to_dict()
    index_record_words(record, words)

    total = len(words.keys())
    current = 0
    for key in words.keys():
        #dsref.collection(u'words').document(f'{key}').set({'word': key, 'records': ArrayUnion(words[key])}, merge=True)
        current += 1
        print(f"Indexed {key} ({current}/{total})")

def post_art_from_records():
    records = dsref.collection(u'art').get()
    for r in records:
        record = r.to_dict()
        post("submit", record)

def index_words():
    print('Indexing words...')
    records = dsref.collection(u'art').get()
    words = {}
    for r in records:
        record = r.to_dict()
        index_record_words(record, words)

    total = len(words.keys())
    current = 0
    for key in words.keys():
        dsref.collection(u'words').document(f'{key}').set({'word': key, 'records': ArrayUnion(words[key])}, merge=True)
        current += 1
        print(f"Indexed {key} ({current}/{total})")


if __name__ == '__main__':
    words = sanatize_prompt("<https://blah.com/blah.png> gandalf, riding on /a/ ^ fancy-fancy-pony.").split()
    print(words)

    post_art_from_records()

