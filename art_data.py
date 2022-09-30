from dataclasses import dataclass
import re

authors = dict()
prompts = dict()

@dataclass
class Avatar:
    url: str

    def to_dict(self):
        return {
            'url': self.url
        }

    @staticmethod
    def from_job(job):
        if 'user' in job and 'avatar' in job['user']:
            return Avatar(job['user']['avatar'])

        return None


@dataclass
class Author:
    display_name: str
    mention: str
    id: str
    avatar: Avatar

    def to_dict(self):
        return {
            'display_name': self.display_name,
            'mention': self.mention,
            'id': self.id,
            'avatar': self.avatar.to_dict() if self.avatar is not None else None
        }

    @staticmethod
    def id_from_job(job):
        if 'data' not in job or 'user' not in job['data']:
            return None
        if 'author-id' not in job['data']['user']:
            return None
        id = f"{job['data']['user']['author-id']}"
        if 'discord-message' in job and not id.startswith("d::"):
            id = f"d::{id}"
        return f"{id}"

    @staticmethod
    def from_job(job):
        jobData = job
        if 'data' in job:
            jobData = job['data']
        id = Author.id_from_job(job)
        if id is None:
            return None
        if id in authors:
            return authors[id]

        user = jobData['user']
        username = user['username'] if 'username' in user else 'unknown'
        mention = user['mention'] if 'mention' in user else 'unknown'
        userid = user['author-id'] if 'author-id' in user else 'unknown'
        avatar = Avatar.from_job(jobData)
        author = Author(username, mention, userid, avatar)
        authors[id] = author
        return author

    @staticmethod
    def from_values(id, display_name, mention, avatar):
        if id is None:
            return None
        if id in authors:
            return authors[id]

        author = Author(display_name, mention, id, Avatar(avatar))
        authors[id] = author
        return author

    @staticmethod
    def from_discord_author(author):
        if id is None:
            return None
        if id in authors:
            return authors[id]
        author = Author(author.display_name, author.mention, f'd::{author.id}', Avatar(author.avatar.url))
        authors[author.id] = author
        return author

    @staticmethod
    def from_discord_message(message):
        author = message.author
        if message.author.display_name == "Midjourney Bot":
            author = message.mentions[0]
        return Author.from_discord_author(author)
    @staticmethod
    def from_discord_context(ctx):
        return Author.from_discord_author(ctx.author)


@dataclass
class Prompt:
    prompt: str

    def to_dict(self):
        return {
            'prompt': self.prompt
        }

    @staticmethod
    def from_string(prompt):
        return Prompt(prompt.lower().replace("*", "").split('--')[0].strip())

    @staticmethod
    def from_job(job):
        if 'data' in job:
            job = job['data']
        if 'parameters' not in job or 'prompt' not in job['parameters']:
            print(f"No prompt found in {job}")
            return None
        prompt = job['parameters']['prompt'] if 'parameters' in job and 'prompt' in job['parameters'] else None
        if prompt is None:
            return None

        if prompt in prompts:
            p = prompts[prompt]
        else:
            p = Prompt.from_string(prompt)
            prompts[prompt] = p

        return p

@dataclass
class Parameters:
    prompt: Prompt
    parameters: dict

    @staticmethod
    def from_dict(data):
        prompt = Prompt.from_string(data['prompt'])
        parameters = data
        return Parameters(prompt, parameters)

    def to_dict(self):
        return {
            'prompt': self.prompt.prompt if self.prompt is not None else None,
            'parameters': self.parameters
        }

    @staticmethod
    def from_job(job):
        if 'parameters' in job:
            return Parameters(Prompt.from_job(job), job['parameters'])

        return None

    @classmethod
    def from_string(cls, param):
        splitparams = param.split('--')
        prompt = Prompt.from_string(splitparams[0])
        parameters = dict()
        parameters['prompt'] = param
        parameters['upscaled'] = param.find("upscaled") != -1
        for p in splitparams:
            kvp = p.split(' ', 1)
            key = kvp[0].strip()
            if key == "": break
            value = kvp[1].strip() if len(kvp) > 1 else "True"
            parameters[key] = value
        return Parameters(prompt, parameters)


@dataclass
class Rating:
    author: Author
    like: int = 0
    quality: int = 0

    def to_dict(self):
        return {
            'author': self.author.to_dict() if self.author is not None else None,
            'like': self.like,
            'quality': self.quality
        }

@dataclass
class Art:
    id: str
    url: str
    author: Author
    parameters: Parameters
    model: str
    width: int
    height: int
    timestamp: float
    aspect_ratio = property(lambda self: self.width / float(self.height) if self.width is not None and self.height is not None else None)

    def to_ref(self):
        return {'id': self.id, 'url': self.url}

    def to_dict(self):
        return {
            'id': self.id,
            'url': self.url,
            'author': self.author.to_dict() if self.author is not None else None,
            'parameters': self.parameters.to_dict() if self.parameters is not None else None,
            'model': self.model,
            'width': self.width,
            'height': self.height,
            'aspect_ratio': self.aspect_ratio,
            'timestamp': self.timestamp
        }

    @staticmethod
    def from_job(job):
        jobData = job['data']
        if 'parameters' not in jobData:
            return

        arts = []
        id = 0
        if 'images' in jobData:
            for image in jobData['images']:
                author = Author.from_job(job)
                model = 'stable-diffusion'
                if 'model' in jobData['parameters']:
                    model = jobData['parameters']['model']
                arts.append(Art(
                    f"{job['name']}::{id}",
                    image,
                    author,
                    Parameters.from_job(jobData),
                    model,
                    int(jobData['parameters']['width']),
                    int(jobData['parameters']['height']),
                    float(job['job']['timestamp']) if 'job' in job and 'timestamp' in job['job'] else None
                ))
                id += 1
        return arts

    @staticmethod
    def from_midjourney_message(message):
        arts = []

        for attachment in message.attachments:
            arts.append(Art.from_midjourney_attachment(message, attachment))

        if len(arts) > 0:
            id = 0
            for art in arts:
                art.id = f"{art.id}::{id}"
                id += 1

        return arts


    @staticmethod
    def from_midjourney_attachment(message, attachment):
        author = Author.from_discord_message(message)
        prompt = message.content
        result = re.search('\*\*(.*)\*\*', prompt)
        if result is not None:
            prompt = result.group(1)

        return Art(
            f'd::{message.id}',
            attachment.url,
            author,
            Parameters.from_string(prompt),
            'midjourney',
            -1,
            -1,
            message.created_at.timestamp())

