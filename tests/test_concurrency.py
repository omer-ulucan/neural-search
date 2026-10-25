import asyncio
import threading
import time

import httpx

from src.inference import LocalLLM
from src.main import Resources, create_app


class SlowGraph:
    def invoke(self, state):
        time.sleep(0.5)
        return {**state, "routing": "local", "response": "done"}


def test_search_does_not_block_the_event_loop(seeded_retriever):
    app = create_app(lambda: None)
    app.state.resources = Resources(retriever=seeded_retriever, graph=SlowGraph())

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            finished = []

            async def call(method, url, **kw):
                await getattr(c, method)(url, **kw)
                finished.append(url)

            search = asyncio.create_task(call("post", "/search", json={"query": "q"}))
            await asyncio.sleep(0.05)
            await call("get", "/health")
            await search
            return finished

    assert asyncio.run(scenario()) == ["/health", "/search"]


def test_llm_generation_is_serialized_across_threads():
    active = 0
    peak = 0
    guard = threading.Lock()

    def fake_llama(prompt, **kw):
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with guard:
            active -= 1
        return {"choices": [{"text": "ok"}]}

    llm = object.__new__(LocalLLM)
    llm.llm = fake_llama
    llm._lock = threading.Lock()
    threads = [threading.Thread(target=llm.generate, args=("p",)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 1
