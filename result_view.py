import discord
import aiohttp
import os
import io
import urllib.parse
from urllib.parse import urlparse
from art_gallery_logger import log_message
import emoji

class ResultView(discord.ui.View):
    msg = None

    def __init__(self, job):
        super().__init__()
        self.job = job

    async def show_status(self, status):
        text = f"“{self.job.name}”\n> {self.job.ctx.author.mention}, {status}!"
        await self.send(text)

    async def show_complete(self):
        i = 0
        for child in self.children:
            child.disabled = i == 4
            i += 1

        url = ""
        if 'grid' in self.job.data:
            url = self.job.data['grid']
        elif 'images' in self.job.data:
            url = self.job.data['images'][0]

        name = os.path.basename(urlparse(url).path)

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    message = await self.send(f"“{self.job.name}”\n> {self.job.ctx.author.mention} your task has completed!\n{url}")
                else:
                    data = io.BytesIO(await resp.read())
                    await self.send(f"“{self.job.name}”\n> {self.job.ctx.author.mention} your task has completed!", file=discord.File(data, name))


    async def send(self, message, file=None):
        embed = None
        if 'name' in self.job.data:
            embed = discord.Embed(title="View Job", url=f"https://aiart.doubtech.com/job/{urllib.parse.quote_plus(self.job.data['name'])}", description="View the full state of the job and its results", color=0x00ff40)

        if self.job.notes is not None:
            message += "\n\n>>> " + self.job.notes

        if file is None:
            if self.msg is None:
                self.msg = await self.job.ctx.send(message, view=self, embed=embed)
            else:
                await self.msg.edit(content=message, view=self, embed=embed)
        else:
            if self.msg is None:
                self.msg = await self.job.ctx.send(message, view=self, file=file, embed=embed)
            else:
                await self.msg.edit(content=message, view=self, file=file, embed=embed)
        return self.msg

    async def send_image(self, interaction, index):
        url = self.job.data['images'][index]
        name = os.path.basename(urlparse(url).path)
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    await interaction.response.send_message(f"“{self.job.name}”\n> {self.job.ctx.author.mention} here is image {index + 1}\n{url}")
                data = io.BytesIO(await resp.read())
                response = await interaction.response.send_message(f"“{self.job.name}”\n> {self.job.ctx.author.mention} here is image {index + 1}", file=discord.File(data, name))
                message = response.message

                await log_message(self.job.dbref, message, "stable-diffusion", True,
                                  prompt=self.job.data['parameters']['prompt'])


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
        await self.job.reroll()