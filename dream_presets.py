import asyncio
import functools
import json
import threading
import urllib

import aiohttp
import re

from sdbot_config_manager import config


async def get_data(url, is_json=True):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = "Failed to parse."
            try:
                data = await response.text()
                data = json.loads(data)
                return data
            except:
                print(f"Data was not proper format for json parsing and will be ignored.\n{data}")
                return None


async def execdream(author, displayName, avatar, mention, prompt, preset="Default", width=512, height=512, checkpoint=None):
    # Split the prompt using '--' as a delimiter
    parts = prompt.split("--")

    fields = []
    for i in range(1, len(parts)):
        # get variable name, first set of characters until the first space
        field_name = parts[i].split(" ")[0]
        # get variable value, everything after the first space
        variable_parts = parts[i].split(" ", 1)
        if len(variable_parts) > 1:
            field_value = variable_parts[1]
            fields.append([field_name, field_value])

    if checkpoint is not None:
        fields.append(["checkpoint", checkpoint])

    # Remove all fields from the prompt
    prompt = parts[0]  # Assuming the actual prompt text is before the first '--'

    # Trim the prompt
    prompt = prompt.strip()

    # Get all of the field values and keys. Encode the values as URL params
    field_values = [f"{field[0]}={urllib.parse.quote_plus(field[1].strip())}" for field in fields]

    # Format a request
    url = f"https://api.aiart.doubtech.com/comfyui/dream?token={config['aiart-token']}&preset={preset}&width={width}&height={height}&prompt={urllib.parse.quote_plus(prompt)}&{'&'.join(field_values)}"

    # Add the author, displayName, and avatar to the request
    url += f"&author_id={urllib.parse.quote_plus(author)}&author_displayName={urllib.parse.quote_plus(displayName)}&author_avatar={urllib.parse.quote_plus(avatar)}"

    print(url)

    # Get the response from the request
    return await get_data(url)

# Create a main to call execdream
if __name__ == "__main__":
    result = asyncio.run(execdream(
        "yolanother@gmail.com",
        "Yolan",
        "https://lh3.googleusercontent.com/a/AAcHTte03VBBSRomK9rg6U6mXQy4woaI9cpKcDNE8DxyFWY=s96-c",
        mention="",
        prompt="The great god Zeus --ar 16:9 --node smaug --checkpoint Deliberate v2"))