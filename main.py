import json
import os
import pickle
import re

from firebase_admin import db

import firebase_admin
import replicate as rep
import discord

from urllib.parse import urlparse

from cryptography.fernet import Fernet
from dotenv import load_dotenv
from replicate.exceptions import ModelError, ReplicateError

import getimageinfo

load_dotenv()

bot = discord.Bot()


def load_config():
    if not os.path.exists("sdbot-config.json"):
        exception = "sdbot-config.json not found. Please create it and add your bot token."
        raise FileNotFoundError(exception)
    with open("sdbot-config.json", "r") as f:
        return json.load(f)

config = load_config()
default_app = None
dbref = None
try:
    cred_obj = firebase_admin.credentials.Certificate(config["firebase"])
    default_app = firebase_admin.initialize_app(cred_obj, {
        'databaseURL': config["firebase-url"]
    })
    dbref = db.reference("/")
except ValueError as e:
    print(e)
    pass

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

async def log_prompt(id, author, prompt, url, model, upscaled=False):
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
        record = dbref.child("prompts").child("data").child(prompt.lower()).child(id)
        add_record(record, id, author, prompt, url, model)
        append(dbref.child("prompts").child("list"), prompt.lower(), id)

        if upscaled:
            p = dbref.child("records").child("upscaled").child(id)
            add_record(p, id, author, prompt, url, model, upscaled)

        p = dbref.child("records").child("all").child(id)
        add_record(p, id, author, prompt, url, model, upscaled)

        p = dbref.child("records").child("models").child(model).child("%s" % id)
        add_record(p, id, author, prompt, url, model, upscaled)

userdata = config["userdata"]
print("User data path set to " + userdata)

def save_user_data():
    with open(userdata, "wb") as f:
        pickle.dump(userdata, f)

def load_user_data():
    if not os.path.exists(userdata):
        return dict()
    with open(userdata, "rb") as f:
        try:
            return pickle.load(f)
        except:
            return dict()

userdata = load_user_data()

def get_user_data(user):
    if user not in userdata:
        userdata[user] = dict()
    return userdata[user]

def get_token(userId):
    user = get_user_data(userId)
    encrypt = Fernet(user["key"])
    return encrypt.decrypt(user["token"]).decode()



@bot.slash_command(description="Set your token for replicate.ai")
async def replicate(ctx, *, token):
    user = get_user_data(ctx.author.id)
    if "key" not in user:
        user["key"] = Fernet.generate_key()

    encrypt = Fernet(user["key"])
    user["token"] = encrypt.encrypt(token.encode())
    save_user_data()
    await ctx.respond(content="Your token has been set")

@bot.slash_command(description="Sync past midjourney prompts with the database")
async def sync(ctx, limit=100):
    print (config["sync"])
    if ctx.author.id not in config["sync"]:
        print(f"Sync permission denied: {ctx.author.name} [{ctx.author.id}]")
        await ctx.respond(f"You do not have permission to run a sync.")
        return

    await ctx.respond(f"Syncing...")
    messages = await ctx.channel.history(limit=int(limit)).flatten()
    msg = await ctx.send(content=f"Starting...")
    i = 0
    count = len(messages)
    for message in reversed(messages):
        i += 1
        if message.author.display_name == "Midjourney Bot" and len(message.attachments) > 0:
            try:
                await msg.edit(f"Syncing: [{i}/{count} {int(i/count * 100)}%] {message.content}")
                await log_midjourney(message)
            except Exception as e:
                print(e)
    await msg.edit("Done!")


async def logo(ctx, *, prompt):
    """Generate a logo from a text prompt using the stable-diffusion model"""
    await ctx.respond(f"“{prompt}”\n> Generating...")

    c = rep.Client(api_token=get_token(ctx.author.id))
    model = c.models.get("laion-ai/erlich")
    prediction = model.predict(prompt=prompt)
    image = prediction.first().image

    await ctx.respond(content=f"“{prompt}”\n{image}")

def get_model(model, ctx):
    c = rep.Client(api_token=get_token(ctx.author.id))
    model = c.models.get(model)
    return model

async def basic_prompt(ctx, model, prompt, width, height, init_image = None, upscale: bool=False):
    await ctx.respond(f"“{prompt}”\n> Generating...")
    msg = await ctx.send(content=f"Generating...")

    try:
        if init_image is None:
            for image in get_model(model, ctx).predict(prompt=prompt, width=width, height=height):
                await msg.edit(content=f"“{prompt}”\n{image}")
        else:
            for image in get_model(model, ctx).predict(prompt=prompt, width=width, height=height, init_image=init_image,):
                await msg.edit(content=f"“{prompt}”\n{image}")

        upscaled = None
        if upscale:
            await msg.edit(f"“{prompt}”\n> Upscaling...\n{image}")
            try:
                upscaled = get_model("nightmareai/real-esrgan", ctx).predict(image=image)
            except ReplicateError as e:
                await ctx.respond(content=f"“{prompt}”\n> Upscaling failed: {e}")
            except ModelError as e:
                await ctx.respond(content=f"“{prompt}”\n> Upscaling failed: {e}")

        print(f"“{prompt}”\n{image}")
        if upscaled is not None:
            await log_prompt(msg.id, ctx.author, prompt, upscaled, model)
            await msg.edit(content=f"“Here is your image {ctx.author.mention}!\n{prompt}”\nOriginal: {image}\nUpscaled: {upscaled}")
        else:
            await log_prompt(msg.id, ctx.author, prompt, image, model)
            await msg.edit(content=f"“Here is your image {ctx.author.mention}!\n{prompt}”\n{image}")
    except ReplicateError as e:
        await ctx.respond(content=f"“{prompt}”\n> Generation failed: {e}")
    except ModelError as e:
        await ctx.respond(content=f"“{prompt}”\n> Generation failed: {e}")
    except KeyError:
        await ctx.respond(
            content=f"{prompt}”\n> {ctx.author.mention}, please set your token with `/replicate` in a private message to {bot.user.mention}")

@bot.slash_command(description="Restore the face on an image using the GFPGAN model")
async def restore_face(ctx, *, image):
    """Restore the face of an image using the GFPGAN model"""
    if image == "":
        await ctx.respond(content="Please provide an image")
        return
    await ctx.respond("Restoring...")
    try:
        image = get_model("tencentarc/gfpgan", ctx).predict(img=image)
        await ctx.respond(content=f"“Here is your restored image {ctx.author.mention}!\n{image}")
    except ModelError as e:
        await ctx.respond(content=f"{ctx.author.mention}, we failed to restore your image.\nReason: {e}")
    except KeyError:
        await ctx.respond(
            content=f"{ctx.author.mention}, please set your token with `/replicate` in a private message to {bot.user.mention}")

@bot.slash_command(description="Upscale an image using the Real-ESRGAN model")
async def upscale(ctx, *, image):
    """Upscale an image using the Real-ESRGAN model"""
    if image == "":
        await ctx.respond(content="Please provide an image")
        return
    await ctx.respond("Upscaling...")
    try:
        image = get_model("nightmareai/real-esrgan", ctx).predict(image=image)
        await ctx.respond(content=f"“Here is your upscaled image {ctx.author.mention}!\n{image}")
    except ReplicateError as e:
        await ctx.respond(content=f"{ctx.author.mention}, we failed to upscale your image.\nReason: {e}")
    except ModelError as e:
        await ctx.respond(content=f"{ctx.author.mention}, we failed to upscale your image.\nReason: {e}")
    except KeyError:
        await ctx.respond(
            content=f"{ctx.author.mention}, please set your token with `/replicate` in a private message to {bot.user.mention}")


@bot.slash_command(description="Generate an animated image from a text prompt using the stable-diffusion model")
async def dream_animated(ctx, *, prompt, width=512, height=512):
    await ctx.respond(f"“{prompt}”\n> Generating...")

    try:
        result = get_model("andreasjansson/stable-diffusion-animation", ctx).predict(prompt=prompt, width=width, height=height)
        for i in result:
            print (i)

        print(f"“{prompt}”\n{image}")
        await ctx.respond(content=f"“Here is your image {ctx.author.mention}!\n{prompt}”\n{image}")
    except KeyError:
        await ctx.respond(
            content=f"“{prompt}”\n> {ctx.author.mention}, please set your token with `/replicate` in a private message to {bot.user.mention}")

@bot.slash_command(description="Queue an image to be generated on the shared device pool")
async def queue(ctx, *, prompt, height=512, width=512,  ddim_steps=50, sampler_name="k_lms",\
                toggles=[1, 2, 3], realesrgan_model_name="RealESRGAN", ddim_eta=0.0, n_iter=1, \
                batch_size=1, cfg_scale=7.5, seed='', fp=None, variant_amount=0.0, variant_seed=''):
    data={
        "parameters": {
            "prompt": prompt,
            "height": height,
            "width": width,
            "ddim_steps": ddim_steps,
            "sampler_name": sampler_name,
            "toggles": toggles,
            "realesrgan_model_name": realesrgan_model_name,
            "ddim_eta": ddim_eta,
            "n_iter": n_iter,
            "batch_size": batch_size,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "fp": fp,
            "variant_amount": variant_amount,
            "variant_seed": variant_seed
        }
    }
    dbref.child("jobs").child("queue").push(data)

@bot.slash_command(description="Generate an image from a text prompt using the stable-diffusion model")
async def dream(ctx, *, prompt, width=512, height=512, init_image=None, upscale: bool=False):
    """Generate an image from a text prompt using the stable-diffusion model"""
    await basic_prompt(ctx, "stability-ai/stable-diffusion", prompt, width, height, init_image, upscale)

# Region: Midjourney Logging

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

async def log_midjourney(message):
    if message.author.display_name == "Midjourney Bot" and len(message.attachments) > 0:
        for attachment in message.attachments:
            author = message.mentions[0]
            url = attachment.url
            prompt = message.content
            upscaled = message.content.lower().find("upscaled") != -1
            result = re.search('\*\*(.*)\*\*', prompt)
            prompt = result.group(1)
            model = "midjourney"
            await log_prompt(message.id, author, prompt, url, "midjourney", upscaled)
            if dbref is not None:
                name = sanatize_key(os.path.basename(urlparse(url).path))
                p = dbref.child("midjourney").child(name)
                add_record(p, message.id, author, prompt, url, model, upscaled)

@bot.event
async def on_message(message):
    message = await message.channel.fetch_message(message.id)  # Get Message object from ID
    if message.author.display_name == "Midjourney Bot" and len(message.attachments) > 0:
        await log_midjourney(message)

@bot.event
async def on_raw_reaction_add(payload):
    print("Received reaction from " + payload.member.name)
    # Get the message that was reacted to
    channel = bot.get_channel(payload.channel_id)  # Get Channel object from ID
    message = await channel.fetch_message(payload.message_id)  # Get Message object from ID
    await log_midjourney(message)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    bot.run(os.getenv("DISCORD_TOKEN"))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
