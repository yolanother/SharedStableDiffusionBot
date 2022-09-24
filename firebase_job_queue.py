from abc import abstractmethod
from threading import Thread, Lock
import time

from google.cloud.firestore_v1 import ArrayUnion

from firebase_job_util import FirebaseUpdateEvent, log
from sdbot_config_manager import dsref, dbref
from art_data import Art, Avatar, Author, Prompt, Parameters


doc = dsref.collection(u'art').document(u'artwork')

host_config = {
    'hostname': 'smaug',
}


class FirebaseJobQueue:
    def __init__(self):
        self.busy = False
        self.hostname = host_config['hostname']
        self.jobqueue = []
        self.lock = Lock()
        self.active_job = None

    def data_node(self, job):
        return dbref.child("jobs").child("data").child(job)

    def queue_node(self, job):
        return dbref.child("jobs").child("queue").child(job)

    def monitor_jobs(self):
        print ('Monitoring jobs...')
        dbref.child("jobs").child("queue").listen(self.queue_update)

    def queue_update(self, status):
        ev = FirebaseUpdateEvent(status)
        job = ev.segments[0]
        log(ev)

        if ev.data is None:
            return

        if '/' == ev.path:
            for job in ev.data.keys():
                self.handle_job(job, ev.data[job])
        if ev.segments[-1] == 'worker' and ev.data == self.hostname:
            if job not in self.jobqueue:
                log(f"Received job assignment for {job}.")
                self.jobqueue.append(job)
                if not self.busy:
                    log(f"Not busy, processing job {job}.")
                    self.process_job(job)
                else:
                    log(f"Busy, adding job {job} to queue.")
        elif ev.segments[-1] == 'status':
            self.handle_status(job, ev.data)
        elif len(ev.segments) == 1:
            self.handle_job(ev.segments[-1], ev.data)

    def handle_job(self, job, data):
        log(f"Handling job: {job} => {data}")

        if 'status' in data:
            self.handle_status(job, data['status'])

    def handle_status(self, job, status):
        log("Received status update: " + status)
        if status == 'requesting':
            self.process_request(job)
        if status == 'canceled':
            self.handle_cancel(job)
        if status == 'complete':
            if job == self.active_job:
                self.complete_job(job)
            elif job in self.jobqueue:
                self.cleanup_job(job)

    def update_state(self, job, state):
        self.queue_node(job).update({u'status': state})
        self.data_node(job).child("job").update({u'status': state})
        self.data_node(job).child("job").update({u'timestamp': time.time()})

    def update_availability(self, job):
        self.queue_node(job).child("available-nodes").child(self.hostname).set(not self.busy)

    def process_request(self, job):

        if job is None:
            return

        if not self.busy:
            log(f"Received request. Announcing availability for job {job}.")
            self.update_availability(job)

    def process_job(self, job):
        self.active_job = job
        self.lock.acquire()
        self.busy = True
        self.update_state(job, 'processing')
        self.lock.release()
        self.on_job_started(job)

    @abstractmethod
    def on_job_started(self, job):
        pass

    def cleanup_job(self, job):
        if job in self.jobqueue:
            self.jobqueue.remove(job)

    def next_in_queue(self):
        if len(self.jobqueue) > 0:
            self.process_job(self.jobqueue[0])
        else:
            queued = dbref.child("jobs").child("queue").get()
            for job in queued.keys():
                if job not in self.jobqueue:
                    self.update_availability(job)

    def cancel(self, job):
        self.complete_job(job, 'canceled')

    def complete_job(self, job, status='complete'):
        log(f"Job {job} completed with status {status}.")
        if job == self.active_job:
            self.active_job = None
            self.lock.acquire()
            self.busy = False
            self.update_state(job, status)
            self.lock.release()
            self.cleanup_job(job)
            self.on_job_completed(job)
            self.next_in_queue()
        else:
            self.cleanup_job(job)

    def fail_job(self, job):
        self.complete_job(job, 'failed')

    def handle_cancel(self, job):
        self.update_state(job, 'canceled')
        self.on_job_canceled(job)
        self.cleanup_job(job)
        self.next_in_queue()

    @abstractmethod
    def on_job_canceled(self, job):
        pass

    @abstractmethod
    def on_job_completed(self, job):
        pass


class FirebaseQueueSimulator(FirebaseJobQueue):
    def __init__(self):
        super().__init__()

    def on_job_started(self, job):
        print ('Processing job: %s' % job)
        self.data_node(job).child('extra-data').set({'foo': 'bar'})

    def on_job_completed(self, job):
        print ('Job complete: %s' % job)

    def on_job_canceled(self, job):
        print ('Job canceled: %s' % job)
        self.cancel(job)

if __name__ == '__main__':
    queue = FirebaseQueueSimulator()
    queue.monitor_jobs()
    while True:
        pass