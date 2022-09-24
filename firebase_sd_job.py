import asyncio
import queue
import time
import os

import asyncio
import nest_asyncio

from firebase_job import FirebaseJob
from result_view import ResultView
from database_sync import sync_job
from sdbot_config_manager import dbref


class StableDiffusionFirebaseJob(FirebaseJob):
    message = None
    loop = asyncio.get_event_loop()
    dataqueue = queue.Queue()

    def __init__(self, ctx, data, name=None, notes=None, preferred_worker=None):
        super().__init__(dbref, data, name, preferred_worker)
        self.ctx = ctx
        self.data = data
        self.dbref = dbref
        self.notes = notes
        self.task_registration = None
        self.output_registration = None

    def update_message_id(self, message):
        messageid = message.id
        channelid = message.channel.id
        mention = message.mentions[0].mention
        guild = message.guild.id
        self.data_node().child('job').child('discord-message').set({
            'message-id': messageid,
            'channel-id': channelid,
            'mention': mention,
            'guild': guild
        })

    async def start_job_async(self):
        self.start_job()
        self.result_view = ResultView(ctx=self.ctx, name=self.name, dbref=self.dbref)
        message = await self.result_view.show_status(self.data, "Queued...")
        self.update_message_id(message)

        while self.status == "processing":
            while self.dataqueue.qsize() > 0:
                data = self.dataqueue.get()
                await self.process_data(data)
            await asyncio.sleep(1)


    async def run(self):
        print ("Queuing prompt request %s" % self.name)
        try:
            await self.ctx.respond(f"“Job received...", ephemeral=True)
        except:
            pass
        await self.start_job_async()

    def on_data_updated(self):
        self.dataqueue.put(self.data)

    def on_status_updated(self, status):
        print (f"Status updated: {status}")

    def on_complete(self):
        self.dataqueue.put(self.data_node().get())

    async def process_data(self, data):
        if self.status == 'complete':
            messageid = await self.result_view.show_complete(data)
            self.update_message_id(messageid)