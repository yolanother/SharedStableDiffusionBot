import json
import os

class UserDataCache:

    def __init__(self, filename="user_data.json"):
        self.filename = filename
        # Load data from the JSON file if it exists or initialize an empty dictionary
        self.data = self.load_data()

    def load_data(self):
        """Load data from the JSON file."""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, "r") as f:
                    return json.load(f)
            except:
                return {}
        else:
            return {}

    def save_data(self):
        """Save data to the JSON file."""
        with open(self.filename, "w") as f:
            json.dump(self.data, f)

    def set(self, user_id, key, value):
        """Set a specific key-value pair for a user."""
        user_str_id = str(user_id)
        if user_str_id not in self.data:
            self.data[user_str_id] = {}
        self.data[user_str_id][key] = value
        self.save_data()

    def get(self, user_id, key, default=None):
        """Retrieve a specific value for a key for a user."""
        return self.data.get(str(user_id), {}).get(key, default)

    def remove(self, user_id):
        """Remove data for a specific user."""
        if str(user_id) in self.data:
            del self.data[str(user_id)]
            self.save_data()


# Usage
userdata = UserDataCache()
