import time
from abc import abstractmethod

from google.cloud.firestore_v1 import ArrayUnion

from firebase_job_util import FirebaseUpdateEvent, log
from sdbot_config_manager import dsref, dbref
from art_data import Art, Avatar, Author, Prompt, Parameters

doc = dsref.collection(u'art').document(u'artwork')


class FirebaseJob:
    def __init__(self, dbref, data, name=None, preferred_worker = None):
        self.data = data
        self.datalistener = None
        self.queuelistener = None
        self.status = 'idle'
        self.dbref = dbref
        self.name = name
        self.preferred_worker = preferred_worker

    def data_node(self):
        return dbref.child("jobs").child("data").child(self.name)

    def queue_node(self):
        return dbref.child("jobs").child("queue").child(self.name)

    def start_job(self):
        if self.name is None:
            print(f"DATA: {self.data}")
            syncdata = dbref.child("jobs").child("data").push({'data': self.data})
            self.name = syncdata.key
        else:
            dbref.child("jobs").child("data").child(self.name).set({'data': self.data})
        self.datalistener = self.data_node().listen(self.data_updated)
        self.data_node().child('job').set({'status': 'requesting', 'request-time': time.time(), 'timestamp': time.time(), 'name': self.name})
        self.queue_node().set({'status': 'requesting', 'request-time': time.time(), 'timestamp': time.time(), 'name': self.name})
        self.queuelistener = self.queue_node().listen(self.queue_updated)
        self.status = 'requesting'

    def cancel(self):
        self.status = 'canceled'
        self.queue_node().update({'status': 'canceled'})
        self.on_canceled()

    def data_updated(self, status):
        ev = FirebaseUpdateEvent(status)

        if ev.data is None:
            return

        if ev.path == '/':
            self.data = ev.data
        else:
            d = self.data

            for segment in ev.segments[:-1]:
                d = d[segment]
            d[ev.segments[-1]] = status.data

        self.on_data_updated()

    def queue_updated(self, status):
        ev = FirebaseUpdateEvent(status)
        print(f"Queue Update: {ev}")

        if ev.segments[0] == 'available-nodes' and self.status == 'requesting':
            self.queue(ev.segments[-1])

        if ev.data is None:
            return

        if ev.path == '/':
            self.process_status_update('status' in ev.data and ev.data['status'])
            if 'available-nodes' in ev.data:
                nodes = ev.data['available-nodes']
                for available_node in nodes.keys():
                    if nodes[available_node]:
                        self.queue(available_node)
                        break
        elif ev.path == '/status':
            self.process_status_update(ev.data)

    def queue(self, node):
        self.queue_node().child('status').set('queued')
        if self.preferred_worker:
            node = self.preferred_worker
        self.queue_node().child('worker').set(node)
        self.data_node().child('job').child('worker').set(node)

    def process_status_update(self, status):
        if status != self.status:
            if status == 'complete':
                self.job_complete()
            if status == 'failed':
                self.job_failed()
            if status == 'canceled':
                self.job_canceled()
            self.status = status

            self.on_status_updated(status)

    @abstractmethod
    def on_status_updated(self, status):
        pass

    def job_cleanup(self):
        self.queue_node().delete()
        try:
            if self.queuelistener is not None:
                self.queuelistener.close()
        except:
            log("Error closing queue listener")
        try:
            if self.datalistener is not None:
                self.datalistener.close()
        except:
            log("Error closing data listener")

    def job_complete(self):
        self.job_cleanup()
        self.on_complete()

    def job_canceled(self):
        self.job_cleanup()
        self.on_canceled()

    def job_failed(self):
        self.job_cleanup()
        self.on_failed()

    @abstractmethod
    def on_complete(self):
        pass

    @abstractmethod
    def on_canceled(self):
        pass

    @abstractmethod
    def on_failed(self):
        pass

    @abstractmethod
    def on_data_updated(self):
        pass

    @abstractmethod
    def on_job_started(self):
        pass

class FirebaseJobTest(FirebaseJob):
    def __init__(self, data):
        super().__init__(data)

    def on_job_started(self):
        print('Job started!')

    def on_data_updated(self):
        print(f"Data Update: {self.data}")

    def on_job_complete(self):
        print('Job complete!')

    def on_job_canceled(self):
        print('Job canceled!')

    def on_job_failed(self):
        print('Job failed!')


if __name__ == '__main__':
    job = FirebaseJobTest({'test': 'test1'})
    job.start_job()
    job = FirebaseJobTest({'test': 'test2'})
    job.start_job()
    job = FirebaseJobTest({'test': 'test3'})
    job.start_job()
    job.cancel()
    job = FirebaseJobTest({'test': 'test4'})
    job.start_job()
    job = FirebaseJobTest({'test': 'test5'})
    job.start_job()
    job = FirebaseJobTest({'test': 'test6'})
    job.start_job()
    while job.status != 'complete':
        pass