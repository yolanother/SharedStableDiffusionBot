import json
import os
import pickle

from enum import Enum

from firebase_admin import db, firestore

import firebase_admin
import discord


from dotenv import load_dotenv

load_dotenv()
bot = discord.Bot()


def load_config():
    if not os.path.exists("sdbot-config.json"):
        exception = "sdbot-config.json not found. Please create it and add your bot token."
        raise FileNotFoundError(exception)
    with open("sdbot-config.json", "r") as f:
        return json.load(f)

config = load_config()
default_app = None
dbref = None
dsref = None
try:
    cred_obj = firebase_admin.credentials.Certificate(config["firebase"])
    backend_app = firebase_admin.initialize_app(cred_obj, {
        'databaseURL': config["firebase-url"]
    }, name="backend")
    dbref = db.reference("/", backend_app)
except ValueError as e:
    print(e)
    pass

webdbref = None
try:
    print(config["firebase-web"])
    cred_obj = firebase_admin.credentials.Certificate(config["firebase-web"])
    web_app = firebase_admin.initialize_app(cred_obj, {
        'databaseURL': config["firebase-web-url"]
    }, name='frontend')
    webdbref = db.reference("/", web_app)
    dsref = firestore.client(web_app)
except ValueError as e:
    print(e)
    pass


userdata = config["userdata"]
print("User data path set to " + userdata)

def save_user_data():
    with open(userdata, "wb") as f:
        pickle.dump(userdata, f)

def load_user_data():
    if not os.path.exists(userdata):
        return dict()
    with open(userdata, "rb") as f:
        try:
            return pickle.load(f)
        except:
            return dict()

userdata = load_user_data()