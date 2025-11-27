## libraries 
from xcapi.query import QueryBuilder
from xcapi.client import XenoCantoClient
from xcapi.downloader import Downloader
import os

# getting API key from env
api_key = os.getenv('XENO_CANTO_API_KEY')

# defining query
q = QueryBuilder().group('birds').species('Athene noctua').quality('A').since('2000-01-01').build()

# get recording
client = XenoCantoClient (api_key = api_key)
recordings = client.search (q)

# getting recordings
downloader = Downloader(output_dir="./data")
downloader.download_recordings(recordings)


# getting metadata
downloader = Downloader (output_dir = './data')
downloader._save_metadata(recordings)

