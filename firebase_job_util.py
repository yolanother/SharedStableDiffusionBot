
def log(message, job=None):
    if job is None:
        print("[JOB QUEUE] %s" % message)
    else:
        print("[JOB QUEUE - %s] %s" % (job["name"], message))


class FirebaseUpdateEvent:
    def __init__(self, update=None):
        if update is not None:
            self.data = update.data
            self.event_type = update.event_type
            self.path = update.path
            self.segments = segments = update.path.strip('/').split('/')

    def __str__(self):
        return f"{self.path} ==> {self.event_type} = {self.data})"

    def shift(self, count):
        ev = FirebaseUpdateEvent()
        ev.event_type = self.event_type
        ev.segments = self.segments[count:]
        ev.path = "/" + "/".join(ev.segments)
        ev.data = self.data
        return ev

    def child(self, child):
        ev = FirebaseUpdateEvent()
        ev.event_type = self.event_type
        ev.path = self.path.replace('/' + child, "")
        ev.segments = ev.path.strip('/').split('/')
        if self.data is not None and child in self.data:
            ev.data = self.data[child]
        else:
            ev.data = None
        return ev

