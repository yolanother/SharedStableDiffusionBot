# SharedStableDiffusionBot
A simple bot for running stable diffusion via replicate. Users must enter their own replicate tokens with the /replicate command

## Installation
* pip install -r requirements.txt
* Create a file named .env. This text file will be used to store secrets for your development environment. Paste in the following:
```DISCORD_TOKEN=<your-token>```
* Create a file called sdbot-config.json. This file will be used to store configuration for your bot. It should look like the following:
```
{
    "userdata": "path\\to\\sdbot-config.json"
}
```

## References
[Replicate Guide](https://replicate.com/blog/build-a-robot-artist-for-your-discord-server-with-stable-diffusion)
