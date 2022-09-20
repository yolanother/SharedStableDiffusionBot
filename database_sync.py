import re

from google.cloud.firestore_v1 import ArrayUnion

from sdbot_config_manager import dsref, dbref
from art_data import Art, Avatar, Author, Prompt, Parameters

doc = dsref.collection(u'art').document(u'artwork')

def sync_midjourney_message(message):
    author = Author.from_discord_message(message)
    arts = Art.from_midjourney_message(message)
    for art in arts:
        dsref.collection(u'art').document(art.id).set(art.to_dict())
        dsref.collection(u'prompts').document(art.parameters.prompt.prompt.replace('/', '')).set({u'images': ArrayUnion([art.to_ref()])}, merge=True)
        dsref.collection(u'models').document(art.model).set({u'images': ArrayUnion([art.to_ref()])}, merge=True)
    dsref.collection(u'authors').document(author.id).set(author.to_dict())

def sync_job(job):
    print (f'Syncing {job["name"]}...')
    author = Author.from_job(job)
    try:
        arts = Art.from_job(job)
        images = []
        for art in arts:
            images.append(art.to_ref())
            dsref.collection(u'art').document(art.id).set(art.to_dict())
            dsref.collection(u'models').document(art.model).set({u'images': ArrayUnion([art.to_ref()])}, merge=True)
        prompt = Prompt.from_job(job)
        if len(images) > 0:
            dsref.collection(u'prompts').document(prompt.prompt).set({u'images': ArrayUnion(images)}, merge=True)

        dbref.child("jobs").child("queue").child(job['name']).delete()
        dbref.child("jobs").child("completed").child(job['name']).set(job)
        dsref.collection(u'authors').document(author.id).set(author.to_dict())
    except Exception as e:
        print (e)


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
            print (art.to_dict())
            dsref.collection(u'art').document(f'{art.id}').set(art.to_dict())
            dsref.collection(u'prompts').document(art.parameters.prompt.prompt.replace('/', '')).set({u'images': ArrayUnion([art.to_ref()])}, merge=True)


def sync_jobs():
    jobs = dbref.child("jobs").child("queue").get()
    for name in jobs.keys():
        job = jobs[name]
        sync_job(job)

def flush_queue():
    jobs = dbref.child("jobs").child("queue").get()
    for name in jobs.keys():
        job = jobs[name]
        dbref.child("jobs").child("queue").child(job['name']).delete()
        dbref.child("jobs").child("completed").child(job['name']).set(job)

