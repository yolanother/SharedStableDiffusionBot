from urllib.parse import urlparse
import re
import os

from attr import dataclass

import getimageinfo

def append(node, child, value):
    try:
        child = child.lower()
        c = node.child(child)
        if c is None:
            node.set({child: []})
        values = c.get()
        if values is None:
            c.set([value])
        elif value not in values:
            # If word is in database, add url to list
            values.append(value)
            c.set(values)
    except Exception as e:
        print (f"Failed to append: {e}")

def append_user_info(author, data):
    data['user'] = {
        'username': author.display_name,
        'mention': author.mention,
        'author-id': author.id,
        'avatar': author.avatar.url
    };

def add_record(node, id, author, prompt, url, model, upscaled=False, parameters=None):
    size = getimageinfo.getsizes(url)
    record = {
        "username": author.display_name,
        "mention": author.mention,
        "author-id": author.id,
        "id": id,
        "avatar": author.avatar.url,
        "prompt": prompt,
        "url": url,
        "model": model,
        "upscaled": upscaled,
        "width": size[1][0],
        "height": size[1][1],
        "parameters": parameters
    }
    print(f"Adding record: {record}")
    node.set(record)

def sanatize_key(key):
    return key.replace(" ", "_")\
        .replace(".png", "")\
        .replace("-", "_")\
        .replace("/", "_")\
        .replace("$", "")\
        .replace("[", "")\
        .replace("]", "")\
        .replace("#", "")\
        .replace(".", "")\
        .lower()

async def process_log_message(dbref, id, author, url, prompt, model=None, upscaled=False, parameters=None):
    print(f"Logging {id}")
    result = re.search('\*\*(.*)\*\*', prompt)
    if result is not None:
        prompt = result.group(1)
    await log_prompt(dbref, id, author, prompt, url, model, upscaled, parameters)
    if dbref is not None:
        name = sanatize_key(os.path.basename(urlparse(url).path))
        p = dbref.child(model).child(name)
        add_record(p, id, author, prompt, url, model, upscaled, parameters)

async def log_message(dbref, message, model, upscaled=False, prompt=None, url=None):
    if prompt is None:
        prompt = message.content
    id = message.id
    if url is not None:
        await process_log_message(dbref, id, message.author, url, prompt, model, upscaled)

    else:
        for attachment in message.attachments:
            url = attachment.url
            author = message.mentions[0]
            await process_log_message(dbref, id, author, url, prompt, model, upscaled)

@dataclass
class Avatar:
    url: str

@dataclass
class Author:
    display_name: str
    mention: str
    id: str
    avatar: Avatar

async def log_job(dbref, job, model, upscaled=False):
    if not 'parameters' in job:
        print (f"No parameters in job: {job}")
        return

    if not 'images' in job:
        print (f"No images in job, skipping.")
        return

    username = "unknown"
    mention = "unknown"
    userid = "unknown"
    avatar = Avatar("unknown")
    if 'user' in job['parameters']:
        user = job['parameters']['user']
        username = 'username' in user if user['username'] else 'unknown'
        mention = 'mention' in user if user['mention'] else 'unknown'
        userid = 'author-id' in user if user['author-id'] else 'unknown'
        avatar.url = 'avatar' in user if user['avatar'] else 'unknown'
    author = Author(username, mention, userid, avatar)

    prompt = job['parameters']['prompt']
    for url in job['images']:
        await log_prompt(dbref, job['name'], author, prompt, url, model, upscaled, job['parameters'])

async def log_prompt(dbref, id, author, prompt, url, model, upscaled, parameters=None):
    if dbref is not None:
        id = "%s" % id
        # Iterate over words in prompt
        #trie = dbref.child("prompts").child("trie")
        for word in prompt.lower().split():
            sanitized_word = sanatize_key(word.replace(",", " ")).replace("_", "")
            try:
                append(dbref.child("prompts").child("words"), sanitized_word, id)
                #trie = trie.child(sanitized_word)
                #add_record(trie.child("_values").child(id), id, author, prompt, url, model, upscaled)
            except Exception as e:
                print(f"Couldn't add {sanitized_word}, {e}")
                pass
        record = dbref.child("prompts").child("data").child(sanatize_key(prompt.lower())).child(id)
        add_record(record, id, author, prompt, url, model, parameters)
        append(dbref.child("prompts").child("list"), prompt.lower(), id)

        if upscaled:
            p = dbref.child("records").child("upscaled").child(id)
            add_record(p, id, author, prompt, url, model, upscaled, parameters)

        p = dbref.child("records").child("all").child(id)
        add_record(p, id, author, prompt, url, model, upscaled, parameters)

        p = dbref.child("records").child("models").child(model).child("%s" % id)
        add_record(p, id, author, prompt, url, model, upscaled, parameters)
        await process_log_message(dbref, id, author, url, prompt, model, upscaled, parameters)