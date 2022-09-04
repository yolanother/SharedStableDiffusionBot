import json
import os
import pickle
import replicate as rep
import discord

from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()

bot = discord.Bot()

def save_user_data():
    with open("sdbot-userdata.pkl", "wb") as f:
        pickle.dump(userdata, f)

def load_user_data():
    if not os.path.exists("sdbot-userdata.pkl"):
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



@bot.slash_command()
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

@bot.slash_command()
async def dream(ctx, *, prompt, width=512, height=512):
    """Generate an image from a text prompt using the stable-diffusion model"""
    msg = await ctx.respond(f"“{prompt}”\n> Generating...")

    try:
        c = rep.Client(api_token=get_token(ctx.author.id))
        model = c.models.get("stability-ai/stable-diffusion")
        result = model.predict(prompt=prompt, width=width, height=height)
        image = result[0]

        print (f"“{prompt}”\n{image}")
        await ctx.respond(content=f"“{prompt}”\n{image}")
    except KeyError:
        await ctx.respond(
            content=f"“{prompt}”\n> Please set your token with `/replicate` in a private message to {bot.user.mention}")
    except Exception as e:
        if "%s" % e == "key" or '%s' % e == "Invalid token.":
            await ctx.respond(content=f"“{prompt}”\n> Please set your token with `/replicate` in a private message to {bot.user.mention}")
        else:
            print(f"“{prompt}”\n> {e}")
            await ctx.respond(content=f"“{prompt}”\n> {e}")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    bot.run(os.getenv("DISCORD_TOKEN"))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
