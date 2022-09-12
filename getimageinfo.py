from urllib import request as ulreq
from PIL import ImageFile
from urllib.request import Request, urlopen  # Python 3

def getsizes(uri):
    # get file size *and* image size (None if not known)
    agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    req = Request(uri)
    req.add_header('User-Agent', agent)
    file = ulreq.urlopen(req)
    size = file.headers.get("content-length")
    if size:
        size = int(size)
    p = ImageFile.Parser()
    while True:
        data = file.read(1024)
        if not data:
            break
        p.feed(data)
        if p.image:
            return size, p.image.size
            break
    file.close()
    return (size, None)