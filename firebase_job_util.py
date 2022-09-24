
def log(message, job=None):
    if job is None:
        print("[JOB QUEUE] %s" % message)
    else:
        print("[JOB QUEUE - %s] %s" % (job["name"], message))

class FirebaseUpdateEvent:
    def __init__(self, update):
        self.data = update.data
        self.event_type = update.event_type
        self.path = update.path
        self.segments = segments = update.path.strip('/').split('/')

    def __str__(self):
        return f"{self.path} ==> {self.event_type} = {self.data})"

