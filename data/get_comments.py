from os import getenv
from time import sleep

from pyyoutube import Api, Comment, CommentThread, PyYouTubeException # pyright: ignore[reportMissingTypeStubs]
from polars import DataFrame, read_avro, concat, col
from requests import ReadTimeout

from utils import Data, Video, read_cached_avro

client = Api(api_key=getenv("YOUTUBE_API_KEY"))
videos = read_avro("videos.avro")
df = read_cached_avro("comments.avro")

def append(df: DataFrame, data: Data) -> DataFrame:
  return concat([df, DataFrame(data.model_dump())], how="vertical", rechunk=True)

def append_comment(df: DataFrame, comment: Comment, video: Video) -> DataFrame:
  snippet = comment.snippet
  if snippet is None:
    raise ValueError("Comment has no snippet")
  comment_id = comment.id
  if comment_id is None:
    raise ValueError("Comment has no ID")
  content = snippet.textDisplay
  if content is None:
    raise ValueError("Comment has no content")
  authorDisplayName = snippet.authorDisplayName
  if authorDisplayName is None:
    raise ValueError("Comment has no author display name")
  authorImageUrl = snippet.authorProfileImageUrl
  if authorImageUrl is None:
    raise ValueError("Comment has no author image URL")
  df = append(df, Data(
    comment_id=comment_id,
    content=content,
    author_name=authorDisplayName,
    author_image_url=authorImageUrl,
    parent_id=snippet.parentId or "",
    video_id=video.video_id,
    video_title=video.video_title,
    video_author=video.video_author
  ))
  df.write_avro("comments.avro")
  return df

def append_commentThreads(df: DataFrame, commentThread: CommentThread, video: Video) -> DataFrame:
  snippet = commentThread.snippet
  if snippet is None:
    raise ValueError("CommentThread has no snippet")
  topLevelComment = snippet.topLevelComment
  if topLevelComment is None:
    raise ValueError("CommentThread has no top-level comment")
  topLevelComment_id = topLevelComment.id
  if topLevelComment_id is None:
    raise ValueError("Top-level comment has no ID")
  df = append_comment(df, topLevelComment, video)
  _replies = commentThread.replies
  replies = (_replies.comments if _replies is not None else []) or []
  if len(replies) == snippet.totalReplyCount:
    for reply in replies:
      df = append_comment(df, reply, video)
  else:
    comments = client.get_comments( # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
      parts="id,snippet",
      parent_id=topLevelComment_id,
    )
    if isinstance(comments, dict):
      raise ValueError(f"Failed to fetch comments for parent ID: {topLevelComment_id}")
    items = comments.items
    if items is None:
      items = []
    for reply in items:
      df = append_comment(df, reply, video)
  return df

for i in videos.iter_rows(named=True):
  video_id: str = i["video_id"] # pyright: ignore[reportAny]
  if len(df) != 0 and df.filter(col("video_id") == video_id).height > 0:
    continue
  print(video_id)
  try:
    comments = client.get_comment_threads( # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
      video_id=video_id,
      count=None,
      parts="replies,snippet,id",
      order="relevance"
    )
  except PyYouTubeException as e:
    if e.message is None:
      raise e
    if e.status_code == 403 and "disabled comments" in e.message:
      print(f"Comments are disabled for video {video_id}. Skipping.")
      continue
    raise e
  except ReadTimeout:
    print("fallback, sleep 10 seconds")
    sleep(10)
    try:
      comments = client.get_comment_threads( # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        video_id=video_id,
        count=None,
        parts="replies,snippet,id",
        order="relevance"
      )
    except ReadTimeout:
      print("Failed to fetch comments after retrying. Skipping video.")
      continue
  if isinstance(comments, dict):
    exit(1)

  items = comments.items
  if not items:
    continue

  for j in items:
    df = append_commentThreads(df, j, Video(
      video_id=video_id,
      video_title=i["video_title"], # pyright: ignore[reportAny]
      video_author=i["video_author"] # pyright: ignore[reportAny]
    ))
