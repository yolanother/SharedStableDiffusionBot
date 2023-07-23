import json
from enum import Enum

from user_settings import get_user_settings


# an enum of settings
class MJSettings(Enum):
    set_settings = "Settings / Set"
    get_settings = "Settings / Get"
    create_preset = "Presets / Create Preset"
    delete_preset = "Presets / Delete Preset"
    list_presets = "Presets / List"
    use_preset = " Presets / Use Preset"
    add_tag = "Tags / Add"
    remove_tag = "Tags / Remove"
    list_tags = "Tags / List"

# A command to set the current mj settings
async def set_mj_settings(ctx, *, value):
    settings = get_user_settings(ctx.author.id)
    settings.set(f"mj_settings", value)
    await ctx.respond(content=f"Your mj commands will now include: {value}.")

# A command to get the current mj settings
async def get_mj_settings(ctx):
    settings = get_user_settings(ctx.author.id)
    settings = settings.get(f"mj_settings", "")
    await ctx.respond(content=f"Your mj commands will include: {settings}.")

# A command to create a preset for mj settings
async def create_mj_preset(ctx, *, name, preset):
    settings = get_user_settings(ctx.author.id)
    settings.set(f"mj_setting_presets::{name}", preset)
    presets = settings.get(f"mj_setting_presets", [])
    presets.remove(name)
    presets.append(name)
    settings.save()
    await ctx.respond(content=f"Your mj preset {name} will now include: {preset}.")

# A command to delete a preset for mj settings
async def delete_mj_preset(ctx, *, name):
    settings = get_user_settings(ctx.author.id)
    presets = settings.get(f"mj_setting_presets", [])
    presets.remove(name)
    settings.delete(f"mj_setting_presets::{name}")
    settings.save()
    await ctx.respond(content=f"Your mj preset {name} has been deleted.")

# A command to get a list of mj presets
async def get_mj_presets(ctx):
    settings = get_user_settings(ctx.author.id)
    presets = settings.get(f"mj_setting_presets", {})
    await ctx.respond(content=f"Your mj presets are: {json.dumps(presets)}.")

def get_preset(ctx, name):
    if name is None or name == "":
        return None

    settings = get_user_settings(ctx.author.id)
    return settings.get(f"mj_setting_presets::{name}", None)

# A command to use a settings preset
async def use_mj_preset(ctx, *, name):
    settings = get_user_settings(ctx.author.id)
    preset = settings.get(f"mj_setting_presets::{name}", "")
    settings.set(f"mj_settings", preset)
    await ctx.respond(content=f"Your mj commands will now include: {preset}.")


# A bot command to add tags
async def tag_add(ctx, *, tag):
    """Add a tag to the database"""
    # get settings from user settings via author's id
    settings = get_user_settings(ctx.author.id)
    tags = settings.get("tags", [])
    tags.append(tag)
    settings.save()
    await ctx.respond(content=f"Added tag {tag} to your settings.")


# A bot command to remove tags
async def tag_remove(ctx, *, tag):
    """Remove a tag from the database"""
    # get settings from user settings via author's id
    settings = get_user_settings(ctx.author.id)
    tags = settings.get("tags", [])
    if tag not in tags:
        await ctx.respond(content=f"Tag {tag} not found in your settings.")
        return
    tags.remove(tag)
    settings.save()
    await ctx.respond(content=f"Removed tag {tag} from your settings.")

# A bot command to list tags
async def tag_list(ctx):
    """List all tags"""
    # get settings from user settings via author's id
    settings = get_user_settings(ctx.author.id)
    tags = settings.get("tags", [])
    await ctx.respond(content=f"```{tags}```")