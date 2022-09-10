import json
import os
import pickle
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
userdata = config["userdata"]
print("User data path set to " + userdata)

def save_user_data():
    with open(userdata, "wb") as f:
        pickle.dump(userdata, f)

def load_user_data():
    if not os.path.exists(userdata):
        return dict()
    with open("sdbot-userdata.pkl", "rb") as f:
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
        if None != upscaled:
            await msg.edit(content=f"“Here is your image {ctx.author.mention}!\n{prompt}”\nOriginal: {image}\nUpscaled: {upscaled}")
        else:
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


@bot.slash_command(description="Generate an image from a text prompt using the stable-diffusion model")
async def dream(ctx, *, prompt, width=512, height=512, init_image=None, upscale: bool=False):
    """Generate an image from a text prompt using the stable-diffusion model"""
    await basic_prompt(ctx, "stability-ai/stable-diffusion", prompt, width, height, init_image, upscale)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    bot.run(os.getenv("DISCORD_TOKEN"))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
