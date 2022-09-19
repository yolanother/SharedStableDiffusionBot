from dataclasses import dataclass

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
        if 'parameters' in job and 'user' in job['parameters'] and 'avatar' in job['parameters']['user']:
            return Avatar(job['parameters']['user']['avatar'])

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
        if 'parameters' not in job:
            return None
        if 'author-id' not in job['parameters']:
            return None
        return job['parameters']['author-id']

    @staticmethod
    def from_job(job):
        id = Author.id_from_job(job)
        if id is None:
            return None
        if id in authors:
            return authors[id]

        user = job['parameters']['user']
        username = 'username' in user if user['username'] else 'unknown'
        mention = 'mention' in user if user['mention'] else 'unknown'
        userid = 'author-id' in user if user['author-id'] else 'unknown'
        avatar = Avatar.from_job(job)
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
            'aspect_ratio': self.aspect_ratio
        }

    @staticmethod
    def from_job(job):
        arts = []
        id = 0
        if 'images' in job:
            for image in job['images']:
                author = Author.from_job(job)
                model = 'stable-diffusion'
                if 'model' in job['parameters']:
                    model = job['parameters']['model']
                arts.append(Art(
                    f"{job['name']}::{id}",
                    image,
                    author,
                    Parameters.from_job(job),
                    model,
                    int(job['parameters']['width']),
                    int(job['parameters']['height'])
                ))
                id += 1
        return arts
