from pyyoutube import Api # pyright: ignore[reportMissingTypeStubs]
from polars import DataFrame
from os import getenv

client = Api(api_key=getenv("YOUTUBE_API_KEY"))
videos = client.get_videos_by_chart( # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
  chart="mostPopular",
  region_code="KR",
  count=None
)

if isinstance(videos, dict):
  exit(1)

items = videos.items
if not items:
  print("No videos found.")
  exit(1)

df: dict[str, list[str]] = {
  "video_id": [],
  "video_title": [],
  "video_author": [],
}

for i in items:
  video_id = i.id
  if not video_id:
    continue
  df["video_id"].append(video_id)
  snippet = i.snippet
  if not snippet:
    continue
  title = snippet.title
  if not title:
    continue
  channel_title = snippet.channelTitle
  if not channel_title:
    continue
  df["video_title"].append(title)
  df["video_author"].append(channel_title)

DataFrame(df).write_avro("videos.avro")

