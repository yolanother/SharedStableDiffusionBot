import asyncio
import queue
import time
import os

import asyncio
import nest_asyncio
from result_view import ResultView

def asyncio_run(future, as_task=True):
    """
    A better implementation of `asyncio.run`.

    :param future: A future or task or call of an async method.
    :param as_task: Forces the future to be scheduled as task (needed for e.g. aiohttp).
    """

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # no event loop running:
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(_to_task(future, as_task, loop))
    else:
        nest_asyncio.apply(loop)
        return asyncio.run(_to_task(future, as_task, loop))


def _to_task(future, as_task, loop):
    if not as_task or isinstance(future, asyncio.Task):
        return future
    return loop.create_task(future)

class FirebaseJob:
    message = None
    loop = asyncio.get_event_loop()
    dataqueue = queue.Queue()

    def __init__(self, ctx, dbref, data, name, notes=None):
        self.ctx = ctx
        self.data = data
        self.dbref = dbref
        self.name = name
        self.notes = notes

    async def reroll(self):
        if 'name' in self.data:
            del self.data['name']
        if 'state' in self.data:
            del self.data['state']
        await self.execute()

    async def execute(self):
        self.processing = True
        self.dbref.child("jobs").child("queue").push(self.data).listen(self.update_status)
        self.result_view = ResultView(self)
        await self.result_view.show_status("Queued...")
        while self.processing:
            if self.dataqueue.qsize() > 0:
                data = self.dataqueue.get()
                await self.process_data(data)
            await asyncio.sleep(1)

    async def run(self):
        print ("Queuing prompt request %s" % self.name)
        await self.ctx.respond(f"“Job received...", ephemeral=True)
        await self.execute()

    def update_status(self, status):
        data = status.data
        self.dataqueue.put(status)

    async def update_queue(self):
        if 'worker' in self.data:
            return
        if 'available-nodes' not in self.data:
            return

        for name in self.data['available-nodes']:
            print (f"{name} is available? {self.data['available-nodes'][name]}")
            self.data['worker'] = name
            self.dbref.child("jobs").child("queue").child(self.data['name']).child('worker').set(name)

    async def process_available_nodes(self, status):
        if 'name' not in self.data:
            return

        name = os.path.basename(status.path)
        if 'available-nodes' not in self.data:
            self.data['available-nodes'] = dict()
        self.data['available-nodes'][name] = status.data
        await self.update_queue()

    async def complete(self):
        await self.result_view.show_complete()
        self.processing = False

    async def process_data(self, status):
        if status.path.startswith("/available-nodes"):
            await self.process_available_nodes(status)
        elif status.path.startswith("/name"):
            self.data['name'] = status.data
        elif status.path.startswith("/state"):
            if status.data == "processing" and 'worker' in self.data:
                await self.result_view.show_status(f"Your job is now being processed by {self.data['worker']}")
            elif status.data == "complete":
                await self.complete()
                self.processing = False
            elif status.data == "error":
                await self.result_view.show_status("Error")
                self.processing = False
        elif status.path == '/':
            self.data = status.data
            print (f"Data updated:  {status.path}\n{status.data}")
            await self.update_queue()
            if "state" in self.data:
                state = self.data['state']
                if state == "processing":
                    await self.result_view.show_status("your task is now being processed!")
                elif state == "complete":
                    await self.complete()