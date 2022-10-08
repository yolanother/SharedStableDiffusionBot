import requests

from art_data import Art
from sdbot_config_manager import config

def post(endpoint, data, *groups):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + config["api-token"],
        'Accept': 'application/json'
    }
    url = config["api-url"] + endpoint
    if len(groups) > 0:
        url += "?"
        for group in groups:
            url += f"groups[]={group}&"

    r = requests.post(url, headers=headers, json=data)
    print(r)
    print(f"Response: {r.text}")

if __name__ == "__main__":
    print("This file is not meant to be run directly.")
    for a in Art.from_job({
        'data': {
            'grid': 'https://doubtech-aiart.s3.amazonaws.com/grid-00262-1203218373_a_badger_wearing_cyberpunk_armor,_highly_detailed_glowing_armor,_scifi_helmet,_tron_style.jpg',
            'images': ['https://doubtech-aiart.s3.amazonaws.com/02179-50_k_lms_1203218373_0.00_a_badger_wearing_cyberpunk_armor,_highly_detailed_glowing_armor,_scifi_helmet,_tron_style.png', 'https://doubtech-aiart.s3.amazonaws.com/02180-50_k_lms_1203218374_0.00_a_badger_wearing_cyberpunk_armor,_highly_detailed_glowing_armor,_scifi_helmet,_tron_style.png', 'https://doubtech-aiart.s3.amazonaws.com/02181-50_k_lms_1203218375_0.00_a_badger_wearing_cyberpunk_armor,_highly_detailed_glowing_armor,_scifi_helmet,_tron_style.png', 'https://doubtech-aiart.s3.amazonaws.com/02182-50_k_lms_1203218376_0.00_a_badger_wearing_cyberpunk_armor,_highly_detailed_glowing_armor,_scifi_helmet,_tron_style.png'],
            'parameters': {'batch_size': 1, 'cfg_scale': 7.5, 'ddim_eta': 0.0, 'ddim_steps': 50, 'height': 512, 'n_iter': 4, 'prompt': 'a badger wearing cyberpunk armor, highly detailed glowing armor, scifi helmet, tron style', 'realesrgan_model_name': 'RealESRGAN', 'sampler_name': 'k_lms', 'seed': '', 'toggles': [1, 2, 3], 'variant_amount': 0.0, 'variant_seed': '', 'width': 768},
            'type': 'txt2img',
            'user': {'author-id': 330453861422596096, 'avatar': 'https://cdn.discordapp.com/avatars/330453861422596096/1effb7a844abba4f567c936a675b7f55.png?size=1024', 'mention': '<@330453861422596096>', 'username': 'Yolan'}},
        'job': {
            'discord-message': {'channel-id': 1006753395631210496, 'guild': 427483707004420097, 'mention': '<@330453861422596096>', 'message-id': 1024138269807620116, 'results-sent': True},
            'name': '-NCwepUP1jRRK_odjOWS',
            'request-time': 1664243947.6124728,
            'status': 'complete',
            'timestamp': 1664243991.294982,
            'worker': 'gandalf'},
        'name': '-NCwepUP1jRRK_odjOWS'}):
        post("submit", a.to_dict())
    exit(1)