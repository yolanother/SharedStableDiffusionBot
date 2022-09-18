from urllib.parse import urlparse
import re
import os

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

def add_record(node, id, author, prompt, url, model, upscaled=False):
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
        "height": size[1][1]
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

async def process_log_message(dbref, id, author, url, prompt, model=None, upscaled=False):
    print(f"Logging {id}")
    result = re.search('\*\*(.*)\*\*', prompt)
    if result is not None:
        prompt = result.group(1)
    await log_prompt(dbref, id, author, prompt, url, model, upscaled)
    if dbref is not None:
        name = sanatize_key(os.path.basename(urlparse(url).path))
        p = dbref.child(model).child(name)
        add_record(p, id, author, prompt, url, model, upscaled)

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


async def log_prompt(dbref, id, author, prompt, url, model, upscaled=False):
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
        add_record(record, id, author, prompt, url, model)
        append(dbref.child("prompts").child("list"), prompt.lower(), id)

        if upscaled:
            p = dbref.child("records").child("upscaled").child(id)
            add_record(p, id, author, prompt, url, model, upscaled)

        p = dbref.child("records").child("all").child(id)
        add_record(p, id, author, prompt, url, model, upscaled)

        p = dbref.child("records").child("models").child(model).child("%s" % id)
        add_record(p, id, author, prompt, url, model, upscaled)