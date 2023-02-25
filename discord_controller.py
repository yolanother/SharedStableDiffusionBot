import pyperclip
import pywinauto.keyboard as keyboard
import random
import time

from pywinauto.application import Application

def SendText(window, text):
    pyperclip.copy(text)
    keyboard.send_keys("^v")
    keyboard.send_keys("{ENTER}")

def SendCommand(window, command, prompt):
    window.set_focus()
    time.sleep(1)
    pyperclip.copy(f'/{command}')
    keyboard.send_keys("^v")
    time.sleep(1.5)
    keyboard.send_keys("{TAB}")
    time.sleep(1)
    keyboard.send_keys("{SPACE}")

    pyperclip.copy(f'{prompt}')
    window.set_focus()
    keyboard.send_keys("^v")
    time.sleep(random.uniform(0, 1) * 3 + .5)
    keyboard.send_keys("{ENTER}")

def GetDiscordWindow(channel="midjourney-tagged-prompts"):
    app = Application(backend="uia").connect(title_re=".*Discord")
    window = app.window(title_re=".*Discord")
    window.set_focus()

    keyboard.send_keys("^k%s{ENTER}" % (channel))
    keyboard.send_keys("{TAB}")

    return window
