import asyncio
import json
import os
import traceback
import urllib
from queue import Queue
from table2ascii import table2ascii as t2a, PresetStyle

from enum import Enum

import chatgpt
import discord_job_cache
import dream_presets
import userdatacache
from discord_job_cache import store_to_cache, cache_job
from mj_commands import MJSettings

import aiohttp
import requests
from firebase_admin import db

import mj_commands

from database_sync import sync_midjourney_message, sync_job, sync_job_by_name
from firebase_job_util import FirebaseUpdateEvent
from firebase_sd_api_job import StableDiffusionFirebaseApiJob, data_listener
from firebase_sd_job import FirebaseJob, StableDiffusionFirebaseJob
from art_gallery_logger import log_prompt, append_user_info, log_job
from art_gallery_logger import log_message

import firebase_admin
import replicate as rep
import discord


import re
import random

from discord import Interaction
from discord.ui import Button, View


from cryptography.fernet import Fernet
from dotenv import load_dotenv
from replicate.exceptions import ModelError, ReplicateError

from result_view import ResultView, JobRunner
from sdbot_config_manager import dbref, userdata, config, bot, save_user_data, webdbref
from user_settings import get_user_settings

completedJobs = dict()

# Region Setting Enumbs

class ChatGptCommands(Enum):
    set_typeblock = "set_typeblock"
    set_prefix = "set_prefix"
    set_postfix = "set_postfix"
    set_default_namespace = "set_default_namespace"
    set_default_agent = "set_default_agent"

class ChatGptHistoryCommands(Enum):
    clear = "clear"
    list = "list"

# End Region

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

    data = append_user_info(ctx.author, data)
    job = StableDiffusionFirebaseJob(ctx=ctx, data=data, preferred_worker=node)
    await job.run()

@bot.slash_command(description="Generate an image from a text prompt using the stable-diffusion model")
async def dream(ctx, *, prompt, width=512, height=512, init_image=None, upscale: bool=False):
    """Generate an image from a text prompt using the stable-diffusion model"""
    await basic_prompt(ctx, "stability-ai/stable-diffusion", prompt, width, height, init_image, upscale)

async def get_data(url, is_json=True):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            try:
                if is_json:
                    data = await response.json()
                else:
                    data = await response.text()
                return data
            except:
                print ("Data was not proper format for json parsing and will be ignored.")
                return None
async def post_data(url, data, is_json=True):
    async with aiohttp.ClientSession() as session:
        print(f"Sending: to {url} with Data:\n{data}")
        headers = {'Content-Type': 'application/json'}
        async with session.post(url, json=data, headers=headers) as response:
            try:
                if is_json:
                    data = await response.json()
                else:
                    data = await response.text()
                return data
            except:
                print ("Data was not proper format for json parsing and will be ignored.")
                return None





async def embellish_prompt(prompt: str, namespace = "aiart", agent = "midjourney"):
    # split the prompt on the first -- and use the first part as the prompt
    split = prompt.split("--", 1)
    processed_prompt = split[0]
    # get the second part including the -- if the prompt was split
    parameters = ""
    if len(split) > 1:
        parameters = " --" + split[1]

    url = f"https://api.aiart.doubtech.com/gpt/agent?token={config['aiart-token']}&namespace={namespace}&agent={agent}&prompt={urllib.parse.quote(processed_prompt)}"
    data = await get_data(url)
    if 'content' in data:
        return data['content'] + parameters

    return prompt

lora_list = None
preset_list = None
checkpoint_list = None


@bot.slash_command(description="Get a list of available checkpoints")
async def dream_checkpoints(ctx):
    global checkpoint_list
    # Get the list of checkpoints from https://api.aiart.doubtech.com/comfyui/list-checkpoints
    url = f"https://api.aiart.doubtech.com/comfyui/checkpoints"
    checkpoint_list = await get_data(url)

    # Convert the list of checkpoints to a table format
    output = t2a(
        header=["Index", "Checkpoint"],
        body=[[i + 1, checkpoint] for i, checkpoint in enumerate(checkpoint_list)],
        style=PresetStyle.thin_compact
    )

    await ctx.respond(content=f"```\n{output}\n```")

@bot.slash_command(description="Get a list of loras")
async def dream_lora_list(ctx):
    url = f"https://api.aiart.doubtech.com/comfyui/list-loras"
    global lora_list
    lora_list = await get_data(url)
    print(f"{lora_list}")
    output_data = []

    for item in lora_list:
        print(item)
        output_data.append([item["name"], ",".join(item["nodes"])])
    output = t2a(
        header=["Lora", "Nodes"],
        body=output_data,
        style=PresetStyle.thin_compact
    )

    await ctx.respond(content=f"```\n{output}\n```")


@bot.slash_command(description="Get a list of dream presets")
async def dream_preset_list(ctx):
    global preset_list
    # Get the list of presets from https://api.aiart.doubtech.com/comfyui/list-presets
    url = f"https://api.aiart.doubtech.com/comfyui/list-presets"
    preset_list = await get_data(url)

    # Convert the list of presets to a table format
    output = t2a(
        header=["Index", "Preset"],
        body=[[i + 1, preset] for i, preset in enumerate(preset_list)],
        style=PresetStyle.thin_compact
    )

    await ctx.respond(content=f"```\n{output}\n```")

async def getpresets(ctx: discord.AutocompleteContext):
    global preset_list
    if preset_list is None:
        url = f"https://api.aiart.doubtech.com/comfyui/list-presets"
        preset_list = await get_data(url)
    return sorted(preset_list)


async def getloras(ctx: discord.AutocompleteContext):
    global lora_list
    if lora_list is None:
        url = f"https://api.aiart.doubtech.com/comfyui/list-loras"
        lora_list = await get_data(url)

    output_data = []

    for item in lora_list:
        print(item)
        output_data.append(item["name"])
    return sorted(output_data)


async def getcheckpoints(ctx: discord.AutocompleteContext):
    global checkpoint_list
    if checkpoint_list is None:
        url = f"https://api.aiart.doubtech.com/comfyui/checkpoints"
        checkpoint_list = await get_data(url)
    return sorted(checkpoint_list)


async def getsdxlpresets(ctx: discord.AutocompleteContext):
    global preset_list
    if preset_list is None:
        url = f"https://api.aiart.doubtech.com/comfyui/list-sdxl-presets"
        preset_list = await get_data(url)
    return sorted(preset_list)

async def getsdxlsizes(ctx: discord.AutocompleteContext):
    global preset_list
    if preset_list is None:
        url = f"https://api.aiart.doubtech.com/comfyui/list-sdxl-sizes"
        preset_list = await get_data(url)
    return sorted(preset_list)


async def executepreset(ctx, *, prompt, preset="Default", width=512, height=512, checkpoint=""):
    """Generate an image from a text prompt using a Stable Diffusion preset"""
    await ctx.respond("Processing...", delete_after=0)

    if "--loras" not in prompt:
        loras = dream_presets.get_loras_string(userdatacache.userdata.get(ctx.author.id, "loras", {}))
        if loras != "":
            prompt += f" --loras {loras}"

    # if preset is empty set it to Default
    if preset == "":
        preset = "Default"

    config = f"Has requested a Stable Diffusion image using the {preset} preset."
    if checkpoint is not None and checkpoint != "":
        config = f"Has requested a Stable Diffusion image using the {preset} preset using the {checkpoint} checkpoint."

    # If the checkpoint is not none get the string value of the Option
    if str(checkpoint) == '':
        checkpoint = None

    # Get the prompt before the first "--" and strip it of whitespace
    promptText = prompt.split("--", 1)[0].strip()

    # Split the prompt by "--" and process the options
    options = ""
    for option in prompt.split("--")[1:]:
        # Split each option into optionname and optionvalue
        option_parts = option.strip().split(" ", 1)
        optionname = option_parts[0]
        optionvalue = option_parts[1] if len(option_parts) > 1 else None

        # Skip the value refiner if it's the same as the value of promptText
        if optionvalue is not None and optionvalue.strip() == promptText:
            continue

        # Append option to the options string
        options += f"* {optionname} => {optionvalue}\n" if optionvalue else f"* {optionname}\n"

    # Add the "Options" header if options are present
    if options:
        options = f"Options:\n{options}\n"

    message = await ctx.send(
        content=f"**{ctx.author.display_name}** {config}\n```{promptText}```\n{options}Status: Your prompt has been queued...",
        view=RespinButtonView(ctx, prompt, preset, width, height, checkpoint))
    try:
        result = await dream_presets.execdream(f'd::{ctx.author.id}', ctx.author.name, ctx.author.avatar.url,
                                               ctx.author.mention, prompt, preset, width, height, checkpoint, loras=userdatacache.userdata.get(ctx.author.id, "loras", {}))
        if result is None:
            await message.edit(
                content=f"{ctx.author.mention}\n```{prompt}```\n\nGeneration with {preset} failed! Error: Unknown")
        elif 'error' in result:
            await message.edit(
                content=f"{ctx.author.mention}\n```{prompt}```\n\nGeneration with {preset} failed! Error: {result['error']}")
            return
        else:
            cache_job(result['id'], message)
    except Exception as e:
        await message.edit(
            content=f"{ctx.author.mention}\n```{prompt}```\n\nGeneration with {preset} failed! Error: {e}")


class Variations(discord.ui.Modal):
    def __init__(self, ctx, prompt, width, height, preset, checkpoint):
        super().__init__(title="Prompt")
        self.prompt = prompt
        self.width = width
        self.height = height
        self.preset = preset
        self.checkpoint = checkpoint
        self.ctx = ctx
        self.prompt_dialog = discord.ui.InputText(
            label="Prompt",
            value=self.prompt,
            placeholder="Enter prompt",
            style=discord.InputTextStyle.paragraph
        )
        self.add_item(self.prompt_dialog)

    async def callback(self, interaction: Interaction):
        await interaction.response.defer()
        await executepreset(self.ctx, prompt=self.prompt_dialog.value, preset=self.preset, width=self.width,
                            height=self.height, checkpoint=self.checkpoint)

    async def on_submit(self, interaction: discord.Interaction):
        await executepreset(self.ctx, prompt=self.prompt_dialog.value, preset=self.preset, width=self.width,
                            height=self.height, checkpoint=self.checkpoint)


class RespinButtonView(View):
    def __init__(self, ctx, prompt, preset, width, height, checkpoint):
        super().__init__()
        self.ctx = ctx
        self.prompt = prompt
        self.preset = preset
        self.width = width
        self.height = height
        self.checkpoint = checkpoint

    # Use a dice emoji for the button
    @discord.ui.button(label='Re-roll', emoji='🎲')
    async def respin_button(self, button: Button, interaction: Interaction):
        await interaction.response.defer()

        # if prompt has --seed <seed> in it, replace it with --seed <random seed>, otherwise append --seed <random seed>
        if '--seed' in self.prompt:
            self.prompt = re.sub(r'--seed \d+', '--seed ' + str(random.randint(0, 999999999)), self.prompt)
        else:
            self.prompt += ' --seed ' + str(random.randint(0, 999999999))

        await executepreset(self.ctx, prompt=self.prompt, preset=self.preset, width=self.width, height=self.height, checkpoint=self.checkpoint)

    @discord.ui.button(label='Variant', emoji='✏️')
    async def variant_button(self, button: Button, interaction: Interaction):
        modal = Variations(ctx=self.ctx, prompt=self.prompt, preset=self.preset, width=self.width, height=self.height, checkpoint=self.checkpoint)
        await interaction.response.send_modal(modal)

    # Use an emoji for edit for the button
    #@discord.ui.button(label='Variant', emoji='✏️')
    #async def variant_button(self, button: Button, interaction: Interaction):
    #    modal = VariantModal(title="Variant")
    #    await self.ctx.send_modal(modal)

@bot.slash_command(description="Generate an image from a text prompt using a Stable Diffusion preset")
async def dream_preset(ctx, *, prompt, preset=discord.Option(default="", autocomplete=getpresets), width=512, height=512, checkpoint=discord.Option(default="", autocomplete=getcheckpoints)):
    if preset is None or preset == "":
        preset = userdatacache.userdata.get(ctx.author.id, "dreampreset", "Default")
    userdatacache.userdata.set(ctx.author.id, "dreampreset", preset)

    await executepreset(ctx, prompt=prompt, preset=str(preset), width=width, height=height, checkpoint=str(checkpoint))


@bot.slash_command(description="Add a lora to your dream requests")
async def dream_add_lora(ctx, *, lora=discord.Option(autocomplete=getloras), model_strength=1, prompt_strength=1):
    if lora is None or lora == "":
        await ctx.respond(content="Please provide a lora")
        return
    loras = userdatacache.userdata.get(ctx.author.id, "loras", {})
    loras[lora] = {"model_strength": model_strength, "prompt_strength": prompt_strength}
    userdatacache.userdata.set(ctx.author.id, "loras", loras)
    await ctx.respond(content=f"Added {lora} to your dream requests")


@bot.slash_command(description="Remove a lora from your dream requests")
async def dream_remove_lora(ctx, *, lora=discord.Option(autocomplete=getloras)):
    if lora is None or lora == "":
        await ctx.respond(content="Please provide a lora")
        return
    loras = userdatacache.userdata.get(ctx.author.id, "loras", {})
    if lora in loras:
        del loras[lora]
        userdatacache.userdata.set(ctx.author.id, "loras", loras)
        await ctx.respond(content=f"Removed {lora} from your dream requests")
    else:
        await ctx.respond(content=f"You don't have {lora} in your dream requests")


@bot.slash_command(description="Get a list of your active loras")
async def dream_get_active_loras(ctx: discord.AutocompleteContext):
    loras = userdatacache.userdata.get(ctx.author.id, "loras", {})
    await ctx.respond(content=f"{ctx.author.mention} your active loras are set to the following:\n```json\n{json.dumps(loras)}\n```")


@bot.slash_command(description="Generate an image using SDXL")
async def sdxl(ctx, *, prompt, refiner=None, negative=None, size=discord.Option(default="1024x1024", autocomplete=getsdxlsizes), preset=discord.Option(default="", autocomplete=getsdxlpresets), width=1024, height=1024, ):
    # Strip anything after and including -- from the prompt and store it in stripped_prompt
    stripped_prompt = prompt.split("--")[0].strip()

    # if prompt doesn't contain --refiner add refiner
    if "--refiner" not in prompt:
        prompt += " --refiner " + (refiner if refiner is not None else stripped_prompt)

    # if prompt doesn't contain --negative add negative
    if "--negative" not in prompt and negative is not None:
        prompt += " --negative " + negative

    # if --width value exists set width to it and remove it from prompt
    if "--width" in prompt:
        width = int(re.search(r'--width (\d+)', prompt).group(1))
        prompt = re.sub(r'--width \d+', '', prompt)

    # if --height value exists set height to it and remove it from prompt
    if "--height" in prompt:
        height = int(re.search(r'--height (\d+)', prompt).group(1))
        prompt = re.sub(r'--height \d+', '', prompt)


    if preset is None or preset == "":
        preset = userdatacache.userdata.get(ctx.author.id, "sdxlpreset", "SDXL")
    userdatacache.userdata.set(ctx.author.id, "sdxlpreset", preset)
    await executepreset(ctx, prompt=prompt, preset=preset, width=width, height=height)

# A bot command to print the chatgpt history for an agent

@bot.slash_command(description="Generate a prompt from a simple prompt")
async def agent(ctx, *, prompt, namespace="", agent=""):
    if namespace == "":
        settings = get_user_settings(ctx.author.id)
        namespace = settings.get(f"chatgpt_default_namespace", "default")
    if agent == "":
        settings = get_user_settings(ctx.author.id)
        agent = settings.get(f"chatgpt_default_agent", "default")

    await chatgpt(ctx, prompt=prompt, namespace=namespace, agent=agent, use_history=False, append_history=False)

# A bot command to generate a prompt
@bot.slash_command(description="Generate a prompt from a simple prompt")
async def chatgpt(ctx, *, prompt, namespace="", agent="", use_history=True, append_history=True):
    """Generate a prompt from a simple prompt"""
    await ctx.respond("Processing...", delete_after=0)

    # get settings from user settings via author's id
    settings = get_user_settings(ctx.author.id)
    agent, namespace = await get_agent(ctx, agent, namespace)

    message = await ctx.send(content=f"**{ctx.author.name}**```{prompt}```\nChat gpt is typing...")

    settings = get_user_settings(ctx.author.id)
    history = []
    if use_history:
        history = settings.get(f"chatgpt_history::{namespace}::{agent}", [])
    else:
        # create a new history array from the old one
        h = []
        for entry in history:
            if entry['role'] == "user":
                h.append(entry)
        history = h

    history.append({"role": "user", "content": prompt})

    # limit history entries to the last 10
    history = history[-10:]

    url = f"https://api.aiart.doubtech.com/gpt/agent?token={config['aiart-token']}&namespace={namespace}&agent={agent}"
    body = {"messages": history}
    response = await post_data(url, body)
    if 'content' in response:
        embellished_prompt = response['content']

    print(f"Response: {embellished_prompt}")

    if prompt != embellished_prompt:
        if append_history:
            print("Appending embellished prompt to history")
            history.append({"role": "agent", "content": embellished_prompt})
            settings.save()

        prefix = settings.get(f"chatgpt_prefix::{namespace}::{agent}", "")
        postfix = settings.get(f"chatgpt_postfix::{namespace}::{agent}", "")

        await message.edit(content=f"{ctx.author.mention}\n```{prompt}```\n\n**Chat GPT:**\n{prefix}{embellished_prompt}{postfix}")
    else:
        await message.edit(content=f"Prompt: \n```{prompt}```")


# a command for chatgpt settings
@bot.slash_command(description="View and manage your settings for ChatGPT agents")
async def chatgpt_settings(ctx, *, command: ChatGptCommands = ChatGptCommands.set_typeblock, typeblock="", prefix="", postfix="", namespace="", agent=""):
    if command == ChatGptCommands.set_typeblock:
        await chatgpt_typeblock(ctx, typeblock=typeblock, namespace=namespace, agent=agent)
    elif command == ChatGptCommands.set_prefix:
        await chatgpt_prefix(ctx, prefix=prefix, namespace=namespace, agent=agent)
    elif command == ChatGptCommands.set_postfix:
        await chatgpt_postfix(ctx, postfix=postfix, namespace=namespace, agent=agent)
    elif command == ChatGptCommands.set_default_agent:
        await set_agent(ctx, agent=agent, namespace=namespace)
    elif command == ChatGptCommands.set_default_namespace:
        await set_namespace(ctx, namespace=namespace)
    else:
        await ctx.respond(content=f"Unknown command {command}")

# A command for mj commands
@bot.slash_command(description="A command to change the mj settings")
async def mj_settings(ctx, *, command: MJSettings, preset_name="", settings=""):
    # switch on the command
    if command == MJSettings.set_settings:
        await mj_commands.set_mj_settings(ctx, settings=settings)
    elif command == MJSettings.get_settings:
        await mj_commands.get_mj_settings(ctx)
    elif command == MJSettings.create_preset:
        await mj_commands.create_mj_preset(ctx, name=preset_name, preset=settings)
    elif command == MJSettings.delete_preset:
        await mj_commands.delete_mj_preset(ctx, name=preset_name)
    elif command == MJSettings.list_presets:
        await mj_commands.get_mj_presets(ctx)
    elif command == MJSettings.use_preset:
        await mj_commands.use_mj_preset(ctx, name=preset_name)
    elif command == MJSettings.add_tag:
        await mj_commands.tag_add(ctx, tag=settings)
    elif command == MJSettings.remove_tag:
        await mj_commands.tag_remove(ctx, tag=settings)
    elif command == MJSettings.list_tags:
        await mj_commands.tag_list(ctx)


@bot.slash_command(description="Generate an image from a text prompt using the stable-diffusion model")
async def mj(ctx, *, prompt, embellish: bool=False, namespace="aiart", agent="midjourney", tags="", title="", alt="", preset=""):
    """Generate an image from a text prompt using the stable-diffusion model"""
    await ctx.respond("Processing...", delete_after=0)
    if embellish:
        message = await ctx.send(content=f"Processing your prompt...\n```{prompt}```\nWaiting for embellishment...")
    else:
        message = await ctx.send(content=f"Processing your prompt...\n```{prompt}```")

    channel_id = ctx.channel_id
    guild_id = ctx.guild_id

    channel = bot.get_channel(channel_id)
    server = bot.get_guild(guild_id)

    channel_name = channel.name if channel else "Unknown channel"
    server_name = server.name if server else "Unknown server"

    print("Received command from {0.author} in {1}/{2} com.hammerandchisel.discord://discord.com/channels/{3}/{4}".format(ctx, server_name, channel_name, guild_id, channel_id))

    embellished_prompt = prompt
    if embellish:
        embellished_prompt = await embellish_prompt(prompt, namespace, agent)

    includeMetadata = False
    metadata = {
    }

    if title != "":
        includeMetadata = True
        metadata["title"] = title
    if alt != "":
        includeMetadata = True
        metadata["alt"] = alt
    if tags != "":
        includeMetadata = True
        metadata["tags"] = []
        for tag in tags.split(","):
            # trin spaces on the tag
            tag = tag.strip()
            metadata["tags"].append(tag)
        settings = get_user_settings(ctx.author.id)
        tags = settings.get("tags", [])
        for tag in tags:
            if tag not in metadata["tags"]:
                metadata["tags"].append(tag)

    if includeMetadata:
        embellished_prompt = json.dumps(metadata) + "::.001 " + embellished_prompt
        prompt = json.dumps(metadata) + "::.001 " + prompt

    if embellish:
        await message.edit(content=f"Original Prompt: \n```{prompt}```\n\nEmbellished Prompt: \n```{embellished_prompt}```")
    else:
        await message.edit(content=f"Prompt: \n```{prompt}```")
    if ctx.author.id in config["midjourney"]:
        if embellish:
            prompt = embellished_prompt
        mjsettings = get_user_settings(ctx.author.id).get(f"mj_settings", "")

        preset = mj_commands.get_preset(ctx, preset)
        if preset is not None:
            mjsettings = preset

        if mjsettings != "":
            prompt += " " + mjsettings
        await ctx.respond("Sending to Midjourney!", delete_after=10)
        # send web request with parameters: prompt, channel_id, guild_id, channel
        url = f"{config['midjourney-webapi']}?prompt={urllib.parse.quote(prompt)}&channel_id={channel_id}&guild_id={guild_id}&channel={channel_name}&server={server_name}"
        await get_data(url, False)



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

dataqueue = Queue()
loop = asyncio.get_event_loop()
def async_call(asyncResult):
    loop.call_soon_threadsafe(asyncio.create_task, asyncResult)

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    sync_cached_jobs()
    while True:

        while dataqueue.qsize() > 0:
            data = dataqueue.get()
            try:
                await process_data(data)
            except Exception as e:
                print(e)
                traceback.print_exc()

        await asyncio.sleep(1)

class QueueJobRunner(JobRunner):
    def __init__(self, ctx, data, preferred_worker=None):
        self.ctx = ctx
        self.data = data
        self.preferred_worker = preferred_worker

    async def run(self):
        if 'name' in self.data:
            del self.data['name']
        if 'images' in self.data:
            del self.data['images']
        if 'grid' in self.data:
            del self.data['grid']
        job = StableDiffusionFirebaseJob(ctx=self.ctx, data=self.data, preferred_worker=self.preferred_worker)
        await job.run()

async def process_data(jobData):
    if 'job' in jobData and 'source' in jobData['job'] and jobData['job']['source'] == 'api':
        job = jobData['name']
        if 'job-synced' in jobData['job'] and jobData['job']['job-synced'] == True:
            try:
                dbref.child('jobs').child('data').child(job).delete()
            except Exception as e:
                print(e)
                traceback.print_exc()
        if 'status' in jobData['job'] and jobData['job']['status'] == 'complete':
            sync_job_by_name(jobData['name'])
        else:
            if 'data' in jobData and 'images' in jobData['data']:
                sync_job_by_name(jobData['name'], False)
            StableDiffusionFirebaseApiJob(jobData['name'], jobData).run()
    elif 'job' in jobData and 'discord-message' in jobData['job']:
        discordmsg = jobData['job']['discord-message']
        job = jobData['job']['name']
        status = jobData['job']['status']
        data = jobData['data']
        if 'worker' in jobData['job']:
            worker = jobData['job']['worker']
        else:
            worker = None

        channel = discordmsg['channel-id']
        m = None
        c = bot.get_channel(channel)
        if 'message-id' in discordmsg:
            message = discordmsg['message-id']
            try:
                m = await c.fetch_message(message)
            except discord.errors.NotFound:
                m = None
        mention = discordmsg['mention']
        print(f'discord msg {discordmsg}')
        if 'results-sent' not in discordmsg:
            if status == 'failed':
                result_view = ResultView(c, job, dbref, QueueJobRunner(c, data, worker))
                result_view.msg = m
                msg = await result_view.show_status(jobData, "Failed")
                if msg is not None:
                    dbref.child('jobs').child('data').child(job).child('job').child('discord-message').child('message-id').set(msg.id)
                dbref.child('jobs').child('data').child(job).child('job').child('discord-message').child('results-sent').set(True)
            if status == 'complete' and job not in completedJobs:
                completedJobs[job] = True
                result_view = ResultView(c, job, dbref, QueueJobRunner(c, data, worker))
                result_view.msg = m
                msg = await result_view.show_complete(jobData)
                if msg is not None:
                    dbref.child('jobs').child('data').child(job).child('job').child('discord-message').child('message-id').set(msg.id)
                dbref.child('jobs').child('data').child(job).child('job').child('discord-message').child('results-sent').set(True)
                sync_job_by_name(job)
            elif m is not None:
                result_view = ResultView(c, job, dbref, QueueJobRunner(c, data, worker))
                result_view.msg = m

                if status == 'processing':
                    status = f'Your job is being processed by {worker}! Waiting for first image...'
                    if 'images' in data and len(data['images']) > 0:
                        status = f"Your job is being processed by {worker}! Iteration {len(data['images'])} of {data['parameters']['n_iter']}"
                        sync_job_by_name(job, False)
                    await result_view.show_status(jobData, status)
                else:
                    await result_view.show_status(jobData, f'Your job is currently in the following state: {status}')
        else:
            dbref.child('jobs').child('data').child(job).delete()

def result_handler(ev):
    if ev.path == '/':
        if ev.data is not None:
            if 'source' in ev.data and ev.data['source'] == 'api':
                print("Processing api request...")
                StableDiffusionFirebaseApiJob(ev.data).run()
            else:
                for job in ev.data.keys():
                    jobData = ev.data[job]
                    jobData['name'] = job
                    dataqueue.put(jobData)
    elif ev.segments[0] in data_listener:
        ev.path = ev.path.substring(ev.segments[0].length + 1)
        ev.segments = ev.segments[1:]
        print(f"Updated request for registered child: {ev}")
        data_listener[ev.segments[0]].queue_event(ev)
    elif ev.segments[-1] == 'grid':
        job = ev.segments[0]
        j = dbref.child('jobs').child('data').child(job).get()
        j['name'] = job
        dataqueue.put(j)
    elif ev.segments[-1] == 'images':
        job = ev.segments[0]
        j = dbref.child('jobs').child('data').child(job).get()
        j['name'] = job
        dataqueue.put(j)
    elif ev.segments[-1] == 'job' and 'status' in ev.data:
        job = ev.segments[0]
        j = dbref.child('jobs').child('data').child(job).get()
        j['name'] = job
        dataqueue.put(j)

def node_syncer(ev):
    print(f"Syncing node state: {ev}")
    try:
        if ev.event_type == 'put':
            webdbref.child('jobs').child('nodes').set(ev.data)
    except:
        print("Error syncing nodes")


def queue_job_data_for_processing(job):
    jobData = dbref.child("jobs").child("data").child(job).get()
    if jobData is not None:
        jobData['name'] = job
        dataqueue.put(jobData)


async def sync_job(job_id):
    job = discord_job_cache.get_from_cache(job_id)
    if job is not None:
        try:
            channel = bot.get_channel(job['channel_id'])
            if channel:
                message = await channel.fetch_message(job['message_id'])
                if message is not None:
                    data = await dream_presets.get_data(f"https://api.aiart.doubtech.com/jobs-v2/job-status?id={job_id}")
                    # Replace the line Status: .* with Status: Your prompt is being processed by .*! Waiting for first image...
                    if data is not None and 'status' in data:
                        status = data['status']
                        if status == 'processing':
                            status = f'Your prompt is being processed by {data["processor"]}! Waiting for first image...'
                        elif status == 'failed':
                            status = f'Your prompt failed to process by {data["processor"]}!'
                        elif status == 'complete':
                            status = f'Your prompt has been processed by {data["processor"]}!'
                        content = message.content
                        content = re.sub(r"Status: .*", f"Status: {status}", content)
                        await message.edit(content=content)

                    if data is not None and 'result' in data:
                        if data['result'] is None:
                            content = f"{message.content}"
                            # replace "Your prompt has been queued.." if it is there with "Job completed without creating an image.
                            content = content.replace("Your prompt has been queued...", "Job completed without creating an image.")
                            await message.edit(content=content)
                            discord_job_cache.remove_from_cache(job_id)
                        else:
                            id = data['result']
                            image = await dream_presets.get_data(f"https://api.aiart.doubtech.com/art?id={id}")
                            if image is not None and len(image) > 0 and 'url' in image[0]:
                                url = image[0]['url']
                                print(f"Job completed with image {url}")
                                embed = discord.Embed()
                                embed.set_image(url=url)
                                await message.edit(content=f"{content}", embed=embed)
                                discord_job_cache.remove_from_cache(job_id)

                    #await message.edit(content=f"{ctx.author.mention}\nPreset: {preset}\n```{prompt}```\n\n{result['url']}")
        except:
            print(f"Error syncing job {job_id}")
            discord_job_cache.remove_from_cache(job_id)

def sync_cached_jobs():
    print("Syncing cached jobs...")
    # Iterate over all cached job keys
    keys = discord_job_cache.get_cache().keys()
    # create a new array with keys
    keysnap = []
    for key in keys:
        keysnap.append(key)

    for job in keysnap:
        # Get the job state from https://api.aiart.doubtech.com/jobs-v2/job-status?id={job} via request
        try:
            r = requests.get(f'https://api.aiart.doubtech.com/jobs-v2/job-status?id={job}')
            if r.status_code == 200:
                # If the job is complete, remove it from the cache
                if r.json()['status'] == 'complete':
                    print("Job is complete, removing from cache and updating message.")
                    async_call(sync_job(job))
        except:
            print(f"...Error syncing {job}")




def activity_syncer(result):
    ev = FirebaseUpdateEvent(result)
    print(f"Syncing activity...")
    sync_cached_jobs()

def full_syncer(result):
    ev = FirebaseUpdateEvent(result)
    print(f"Syncing full state...")

    if ev.path == '/':
        data = ev.child('data')
        if data.data is not None:
            keys = data.data.keys()
            for child in keys:
                queue_job_data_for_processing(child)
            node = ev.child('nodes')
            if node is not None and node.data is not None:
                keys = node.data.keys()
                for child in keys:
                    node_syncer({child: node.child(child)})
    elif ev.segments[0] == 'nodes':
        try:
            webdbref.child(f'{ev.path}'.lstrip('/')).set(ev.data)
        except:
            print("Error syncing nodes")
            traceback.print_exc()
    elif len(ev.segments) > 1:
        job = ev.segments[1]
        queue_job_data_for_processing(job)


# REGION CHAT HISTORY

async def chatgpt_show_history(ctx, namespace, agent):
    agent, namespace = await get_agent(ctx, agent, namespace)
    settings = get_user_settings(ctx.author.id)
    history = settings.get(f"chatgpt_history::{namespace}::{agent}", [])

    # limit the number of characters shown for content to 10
    for entry in history:
        # Remove any discord ``` formatting for the entry
        entry["content"] = entry["content"].replace("```", "")

        if len(entry["content"]) > 50:
            entry["content"] = entry["content"][:50] + "..."

    await ctx.respond(content=f"History for {namespace}/{agent}\n```{json.dumps(history, indent=2)}```")

async def chatgpt_clearhistory(ctx, namespace="", agent=""):
    agent, namespace = await get_agent(ctx, agent, namespace)
    settings = get_user_settings(ctx.author.id)
    settings.set(f"chatgpt_history::{namespace}::{agent}", [])
    await ctx.respond(content=f"Your history for {namespace}::{agent} has been cleared.")

# create a command to set a type block for agent responses
async def chatgpt_typeblock(ctx, *, typeblock="", namespace="", agent=""):
    agent, namespace = await get_agent(ctx, agent, namespace)

    settings = get_user_settings(ctx.author.id)
    settings.set(f"chatgpt_prefix::{namespace}::{agent}", f"```{typeblock}\n")
    settings.set(f"chatgpt_postfix::{namespace}::{agent}", f"```")
    await ctx.respond(content=f"Type block set for {namespace}::{agent}.")


async def get_agent(ctx, agent, namespace):
    if namespace == "":
        settings = get_user_settings(ctx.author.id)
        namespace = settings.get(f"chatgpt_default_namespace", "default")
    if agent == "":
        settings = get_user_settings(ctx.author.id)
        agent = settings.get(f"chatgpt_default_agent", "default")
    return agent, namespace


# create a commmand to set the agent's prefix for its responses
async def chatgpt_prefix(ctx, prefix, namespace="", agent=""):
    agent, namespace = await get_agent(ctx, agent, namespace)

    settings = get_user_settings(ctx.author.id)
    settings.set(f"chatgpt_prefix::{namespace}::{agent}", prefix)
    await ctx.respond(content=f"Prefix for {namespace}::{agent} set to {prefix}.")
    settings.save()


async def chatgpt_postfix(ctx, prefix, namespace="", agent=""):
    agent, namespace = await get_agent(ctx, agent, namespace)

    settings = get_user_settings(ctx.author.id)
    settings.set(f"chatgpt_postfix::{namespace}::{agent}", prefix)
    await ctx.respond(content=f"Postfix for {namespace}::{agent} set to {prefix}.")
    settings.save()


# print the default agent
async def default_agent(ctx):
    settings = get_user_settings(ctx.author.id)
    agent = settings.get(f"chatgpt_default_agent", "default")
    namespace = settings.get(f"chatgpt_default_namespace", "default")
    await ctx.respond(content=f"Your default agent is {namespace}/{agent}.")


# print the default namespace
async def default_namespace(ctx):
    settings = get_user_settings(ctx.author.id)
    namespace = settings.get(f"chatgpt_default_namespace", "default")
    await ctx.respond(content=f"Your default namespace is {namespace}.")


# A slash command to set the default agent
async def set_agent(ctx, *, agent="default", namespace=""):
    settings = get_user_settings(ctx.author.id)
    settings.set(f"chatgpt_default_agent", agent)
    if namespace != "":
        settings.set(f"chatgpt_default_namespace", namespace)
    await ctx.respond(content=f"Your default agent is now {agent}.")


# A slash command to set the default namespace
async def set_namespace(ctx, *, namespace="default"):
    settings = get_user_settings(ctx.author.id)
    settings.set(f"chatgpt_default_namespace", namespace)
    await ctx.respond(content=f"Your default namespace is now {namespace}.")


@bot.slash_command(description="View and manage your history for ChatGPT agents")
async def chatgpt_history(ctx, *, command: ChatGptHistoryCommands = ChatGptHistoryCommands.list, namespace="", agent=""):
    if command == ChatGptHistoryCommands.list:
        await chatgpt_show_history(ctx, namespace, agent)
    elif command == ChatGptHistoryCommands.clear:
        await chatgpt_clearhistory(ctx, namespace, agent)
    else:
        await ctx.respond(content=f"Unknown command {command}")


# END REGION CHATGPT HISTORY




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("Listening for updates to data...")
    webdbref.child('activity').listen(activity_syncer)
    dbref.child('jobs').listen(full_syncer)
    dbref.child('activity').listen(activity_syncer)
    print("Connecting to discord...")
    bot.run(os.getenv("DISCORD_TOKEN"))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
