import json

user_settings = {}

class UserSettings:
    def __init__(self, user_id):
        self.user_id = user_id
        self.settings = {}
        try:
            with open(f"data/{user_id}", "r") as f:
                self.settings = json.load(f)
        except FileNotFoundError:
            # If the file does not exist, the settings dictionary will be empty
            pass

    def load(self):
        try:
            with open(f"data/{self.user_id}", "r") as f:
                self.settings = json.load(f)
        except FileNotFoundError:
            # If the file does not exist, the settings dictionary will be empty
            self.settings = {}

    def save(self):
        with open(f"data/{self.user_id}", "w") as f:
            json.dump(self.settings, f, indent=4)

    def set(self, key, value):
        self.settings[key] = value
        self.save()

    def get(self, key, default=None):
        if key not in self.settings:
            self.settings[key] = default
            return default

        return self.settings.get(key)

    def delete(self, key):
        if key in self.settings:
            del self.settings[key]
            self.save()



def get_user_settings(user_id) -> UserSettings:
    if user_id in user_settings:
        return user_settings[user_id]
    user_settings[user_id] = UserSettings(user_id)
    return user_settings[user_id]