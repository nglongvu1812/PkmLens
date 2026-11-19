"""Bound request streams before multipart spooling."""
from starlette.responses import JSONResponse

class BodyLimit:
    def __init__(self, app, maximum=9*1024*1024):
        self.app,self.maximum=app,maximum

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in ("GET","HEAD","OPTIONS"):
            return await self.app(scope,receive,send)
        maximum=17*1024*1024 if scope.get("path","").endswith("/photos") else self.maximum
        chunks=[]
        total=0
        while True:
            event=await receive()
            if event["type"]=="http.disconnect":
                return
            total+=len(event.get("body",b""))
            if total>maximum:
                return await JSONResponse({"detail":"request_too_large"},status_code=413)(scope,receive,send)
            chunks.append(event)
            if not event.get("more_body",False):
                break
        async def bounded_receive():
            if chunks:
                return chunks.pop(0)
            return await receive()
        await self.app(scope,bounded_receive,send)

