import traceback
from abc import abstractmethod

import discord
import aiohttp
import os
import io
import urllib.parse
from urllib.parse import urlparse

class ResultView(discord.ui.View):
    msg = None

    def __init__(self, ctx, name, dbref, reroll_runner=None):
        super().__init__()
        self.ctx = ctx
        self.name = name
        self.data = None
        self.dbref = dbref
        self.reroll_runner = reroll_runner
        
    def prompt(self):
        return self.data['data']['parameters']['prompt']

    def mention(self):
        return self.data['data']['user']['mention']

    async def show_status(self, data, status):
        self.data = data
        text = f"“{self.prompt()}”\n> {self.mention()}, {status}"
        await self.send(text)
        return self.msg

    async def show_complete(self, data):
        self.data = data
        i = 0
        for child in self.children:
            child.disabled = i == 4
            i += 1

        url = ""
        if 'grid' in data['data']:
            url = data['data']['grid']
        elif 'images' in data['data']:
            url = data['data']['images'][0]

        if not url:
            print(f"Called complete without result data. {data}")
            return self.msg

        name = os.path.basename(urlparse(url).path)
        delmsg = self.msg
        if self.msg is not None:
            self.msg = None

        print ("Sending final result...")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        self.msg = message = await self.send(f"“{self.prompt()}”\n> {self.mention()} your task has completed!\n{url}")
                    else:
                        data = io.BytesIO(await resp.read())
                        self.msg = await self.send(f"“{self.prompt()}”\n> {self.mention()} your task has completed!", file=discord.File(data, name))
            print("Done!")
            if delmsg is not None:
                await delmsg.delete()
        except Exception as e:
            print(f"Error sending image: {url}, {e}, {data}")
            traceback.print_exc()

        return self.msg

    async def get_file_attachment(url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                else:
                    data = io.BytesIO(await resp.read())
                    return discord.File(data, name)

    async def send(self, message, file=None):
        embed = None
        if self.prompt() is not None:
            joburl = f"https://aiart.doubtech.com/job/{urllib.parse.quote_plus(self.name)}"
            desc = "View the full state of the job and its results"
            embed = discord.Embed(title="View Job", url=joburl, description=desc, color=0x00ff40)

        if 'parameters' in self.data['data'] is not None:
            options = "Options: "
            for option in self.data['data']['parameters']:
                if option != 'prompt' and self.data['data']['parameters'][option]:
                    options += f"{option}={self.data['data']['parameters'][option]}, "
            options = options[:-2]
            message += f"\n\n>>> {options}"

        if file is None:
            if self.msg is None:
                self.msg = await self.ctx.send(message, view=self, embed=embed)
            else:
                await self.msg.edit(content=message, view=self, embed=embed)
        else:
            if self.msg is None:
                self.msg = await self.ctx.send(message, view=self, file=file, embed=embed)
            else:
                await self.msg.edit(content=message, view=self, file=file, embed=embed)
        return self.msg

    async def send_image(self, interaction, index):
        url = self.data['data']['images'][index]
        name = os.path.basename(urlparse(url).path)
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    await interaction.response.send_message(f"“{self.prompt()}”\n> {self.mention()} here is image {index + 1}\n{url}")
                data = io.BytesIO(await resp.read())
                response = await interaction.response.send_message(f"“{self.prompt()}”\n> {self.mention()} here is image {index + 1}", file=discord.File(data, name))
                message = response.message


    @discord.ui.button(label="U1", style=discord.ButtonStyle.blurple, disabled=True)
    async def upscale_first(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self.send_image(interaction, 0)

    @discord.ui.button(label="U2", style=discord.ButtonStyle.blurple, disabled=True)
    async def upscale_second(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self.send_image(interaction, 1)

    @discord.ui.button(label="U3", style=discord.ButtonStyle.blurple, disabled=True)
    async def upscale_third(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self.send_image(interaction, 2)

    @discord.ui.button(label="U4", style=discord.ButtonStyle.blurple, disabled=True)
    async def upscale_fourth(self, button: discord.ui.Button, interaction: discord.Interaction):
        await self.send_image(interaction, 3)

    @discord.ui.button(label="X", row=1, style=discord.ButtonStyle.red)
    async def delete(self, button: discord.ui.Button, interaction: discord.Interaction):
        pass

    @discord.ui.button(label="Reroll", row=1, emoji="🔄", style=discord.ButtonStyle.gray, disabled=True)
    async def reroll(self, button: discord.ui.Button, interaction: discord.Interaction):
        await interaction.response.send_message("Rerolling!")
        await self.reroll_runner.run()


class JobRunner:
    @abstractmethod
    async def run(self):
        pass