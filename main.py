import json
import os
import pickle

from firebase_admin import db
from firebase_job import FirebaseJob
from art_gallery_logger import log_prompt
from art_gallery_logger import log_message

import firebase_admin
import replicate as rep
import discord


from cryptography.fernet import Fernet
from dotenv import load_dotenv
from replicate.exceptions import ModelError, ReplicateError

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

reserved = dict()

@bot.slash_command(description="Reserve a system for dedicated queuing")
async def reserve(ctx, *, node):
    if ctx.author.id not in config["sync"]:
        print(f"Reserve permission denied: {ctx.author.name} [{ctx.author.id}]")
        await ctx.respond(f"I'm sorry {ctx.author.name}, you do not have permission to reserve a system")
        return
    if node not in reserved:
        reserved[node] = {
            'author': ctx.author,
            'node': node
        }
        await ctx.respond(f"{node} is now reserved for {ctx.author.name}")

@bot.slash_command(description="Releases a system from dedicated queuing")
async def release(ctx, *, node):
    if ctx.author.id not in config["sync"]:
        print(f"Reserve permission denied: {ctx.author.name} [{ctx.author.id}]")
        await ctx.respond(f"I'm sorry {ctx.author.name}, you do not have permission to reserve a system")
        return
    if node in reserved and ctx.author.id == reserved[node].author.id:
        del reserved[node]
        await ctx.respond(f"{node} is now freely available for use")




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
            await log_prompt(dbref, msg.id, ctx.author, prompt, upscaled, model)
            await msg.edit(content=f"“Here is your image {ctx.author.mention}!\n{prompt}”\nOriginal: {image}\nUpscaled: {upscaled}")
        else:
            await log_prompt(dbref, msg.id, ctx.author, prompt, image, model)
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

@bot.slash_command(description="Queue an image to be generated on the shared device pool")
async def queue(ctx, *, prompt, height=512, width=512,  ddim_steps=50, sampler_name="k_lms",\
                toggles=[1, 2, 3], realesrgan_model_name="RealESRGAN", ddim_eta=0.0, n_iter=4, \
                batch_size=1, cfg_scale=7.5, seed='', fp=None, variant_amount=0.0, variant_seed='', node=None):

    if node is not None and node in reserved:
        await ctx.respond(content=f"“{prompt}”\n> {ctx.author.mention}, this node is currently reserved for {reserved[node].display_name}. We'll add you to the queue for the next available machine.")
        node = None
        return

    if ctx.author.id not in config["sync"]:
        n_iter = min(n_iter, 4)

    data={
        "worker": node,
        "type": "txt2img",
        "parameters": {
            "prompt": prompt,
            "height": height,
            "width": width,
            "ddim_steps": min(ddim_steps, 500),
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
    job = FirebaseJob(ctx, dbref, data, prompt)
    await job.run()

@bot.slash_command(description="Generate an image from a text prompt using the stable-diffusion model")
async def dream(ctx, *, prompt, width=512, height=512, init_image=None, upscale: bool=False):
    """Generate an image from a text prompt using the stable-diffusion model"""
    await basic_prompt(ctx, "stability-ai/stable-diffusion", prompt, width, height, init_image, upscale)

async def log_midjourney(message):
    if message.author.display_name == "Midjourney Bot" and len(message.attachments) > 0:
        upscaled = message.content.lower().find("upscaled") != -1
        log_message(dbref, message, "midjourney", upscaled)

@bot.event
async def on_message(message):
    try:
        message = await message.channel.fetch_message(message.id)  # Get Message object from ID
        if message.author.display_name == "Midjourney Bot" and len(message.attachments) > 0:
            await log_midjourney(message)
    except discord.errors.NotFound:
        print(f"Message {message.id} not found, will not log it.")


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
