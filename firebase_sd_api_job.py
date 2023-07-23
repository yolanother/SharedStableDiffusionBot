import asyncio
import queue
import random
import time
import os

import asyncio
import nest_asyncio

from firebase_job import FirebaseJob
from firebase_job_util import FirebaseUpdateEvent
from result_view import ResultView
from database_sync import sync_job
from sdbot_config_manager import dbref

data_listener = dict()

class StableDiffusionFirebaseApiJob:
    loop = asyncio.get_event_loop()
    dataqueue = queue.Queue()

    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.job = data['job']
        self.status = self.job['status']

    def data_node(self):
        return dbref.child("jobs").child("data").child(self.name)

    def queue_node(self):
        return dbref.child("jobs").child("queue").child(self.name)

    def process_full_data_event(self, ev):
        self.job = ev.data
        if 'status' in ev.data:
            self.status = ev.data['status']
            if self.status == 'requesting':
                self.queue_event(ev)
        if self.status == 'complete' and 'name' in self.job:
            sync_job(self.job)

    def data_updated(self, status):
        ev = FirebaseUpdateEvent(status)
        if(ev.data is None):
            return

        if ev.path == '/':
            self.process_full_data(ev.data)
        elif ev.path == "/status":
            self.job['status'] = ev.data
            updateev = FirebaseUpdateEvent()
            updateev.data = self.job
            updateev.path = '/'
            updateev.segments = []
            updateev.event_type = ev.event_type
            self.queue_event(updateev)

    def run(self):
        data_listener[self.name] = self
        ev = FirebaseUpdateEvent()
        ev.path = '/'
        ev.segments = ['']
        ev.event_type = 'put'
        ev.data = self.data
        self.process_full_data_event(ev)
        j = dbref.child('jobs').child('queue').child(self.name).get()
        ev.data = j
        self.queue_event(ev)

    def queue_updated(self, status):
        ev = FirebaseUpdateEvent(status)
        self.queue_event(ev)

    def queue_event(self, ev):
        if ev.segments[0] == 'available-nodes' and self.status == 'requesting':
            self.queue(ev.segments[-1])

        if ev.data is None:
            return

        if ev.path == '/':
            if 'available-nodes' in ev.data:
                nodes = ev.data['available-nodes']

                l = []
                for available_node in nodes.keys():
                    l.append(available_node)
                    if nodes[available_node]:
                        self.queue(available_node)
                        return
                if len(l) > 0:
                    node = l[random.randrange(len(l))]
                    self.queue(node)

    def queue(self, node):
        self.queue_node().child('status').set('queued')
        self.queue_node().child('worker').set(node)
        self.data_node().child('job').child('worker').set(node)
        del data_listener[self.name]
