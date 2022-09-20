import json
import os
import pickle

from enum import Enum

from firebase_admin import db

from database_sync import sync_midjourney_message
from firebase_job import FirebaseJob
from art_gallery_logger import log_prompt, append_user_info, log_job
from art_gallery_logger import log_message

import firebase_admin
import replicate as rep
import discord


from cryptography.fernet import Fernet
from dotenv import load_dotenv
from replicate.exceptions import ModelError, ReplicateError


from sdbot_config_manager import dbref, userdata, config, bot, save_user_data

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

def is_admin(ctx):
    return ctx.author.id in config["sync"]

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

class Sampler(Enum):
    """Enum for the sampler to use"""
    PLMS = "PLMS"
    DDIM = "DDIM"
    k_dpm_2_a = "k_dpm_2_a"
    k_dpm_2 = "k_dpm_2"
    k_euler_a = "k_euler_a"
    k_euler = "k_euler"
    k_heun = "k_heun"
    k_lms = "k_lms"

@bot.slash_command(description="Clean up any orphaned jobs")
async def clean_jobs(ctx, *, model="stable-diffusion"):
    if not is_admin(ctx):
        await ctx.respond(content="You do not have permissions to clean jobs.")
        return

    await ctx.respond(content="Cleaning jobs...")
    count = 0
    jobs = dbref.child("jobs").child("queue").get()
    for name in jobs.keys():
        job = jobs[name]
        print (f'Syncing {job["name"]}...')
        await log_job(dbref, job, model)
        count += 1
        if count > 3:
            break

@bot.slash_command(description="Queue an image to be generated on the shared device pool")
async def queue(ctx, *, prompt, height: int=512, width: int=512,  ddim_steps: int=50, sampler_name: Sampler=Sampler.k_lms, realesrgan_model_name="RealESRGAN", ddim_eta: float=0.0, n_iter: int=4, \
                batch_size: int=1, cfg_scale: float=7.5, seed='', fp=None, variant_amount: float=0.0, variant_seed='', node=None,
                upscale: bool=False, normalize_prompt_weights: bool=True,  fix_faces: bool=False):

    if node is not None and node in reserved:
        await ctx.respond(content=f"“{prompt}”\n> {ctx.author.mention}, this node is currently reserved for {reserved[node].display_name}. We'll add you to the queue for the next available machine.")
        node = None
        return

    toggles = []
    if upscale:
        toggles.append(9)
    if normalize_prompt_weights:
        toggles.append(1)
    if fix_faces:
        toggles.append(8)

    if ctx.author.id not in config["sync"]:
        n_iter = min(n_iter, 4)

    sampler_name = sampler_name.value

    data={
        "worker": node,
        "type": "txt2img",
        "parameters": {
            "prompt": prompt,
            "height": int(height),
            "width": int(width),
            "ddim_steps": min(int(ddim_steps), 500),
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

    options = f"Options: {n_iter} iterations, {width}x{height}, {sampler_name}, {realesrgan_model_name}, {ddim_steps} steps, {ddim_eta} eta, {batch_size} batch size, {cfg_scale} scale, {seed} seed, {fp} fp, {variant_amount} variant amount, {variant_seed} variant seed"

    if upscale:
        options += ", upscaling"
    if normalize_prompt_weights:
        options += ", normalized prompt weights"
    if fix_faces:
        toggles.append(8)
        options += ", fixed faces"

    toggles.append(2)
    toggles.append(3)

    append_user_info(ctx.author, data['parameters'])
    job = FirebaseJob(ctx, dbref, data, prompt, options)
    await job.run()

@bot.slash_command(description="Generate an image from a text prompt using the stable-diffusion model")
async def dream(ctx, *, prompt, width=512, height=512, init_image=None, upscale: bool=False):
    """Generate an image from a text prompt using the stable-diffusion model"""
    await basic_prompt(ctx, "stability-ai/stable-diffusion", prompt, width, height, init_image, upscale)

async def log_midjourney(message):
    if message.author.display_name == "Midjourney Bot" and len(message.attachments) > 0:
        sync_midjourney_message(message)


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
