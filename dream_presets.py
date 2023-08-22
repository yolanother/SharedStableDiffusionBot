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

def get_loras_string(loras):
    loras_string = ""
    # for each key in loras (if any)
    for key in loras:
        # get the {model_strength, prompt_strength} and format all the data into a string: "{key}:{model_strength}:{prompt_strength}"
        lora_string = f"{key}:{loras[key]['model_strength']}:{loras[key]['prompt_strength']}"
        # append the lora_string to the loras_string
        loras_string += lora_string + ","
    if loras_string != "":
        # remove the last comma from the loras_string
        loras_string = loras_string[:-1]
    return loras_string

async def execdream(author, displayName, avatar, mention, prompt, preset="Default", width=512, height=512, checkpoint=None,loras={}):
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

    if "loras" not in fields:
        loras_string = get_loras_string(loras)

        if loras_string != "":
            # add the loras_string to the field_values
            field_values.append(f"loras={urllib.parse.quote_plus(loras_string)}")

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