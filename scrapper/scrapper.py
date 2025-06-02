from googleapiclient.discovery import build
import pandas as pd

api_key = 'API_KEY'
video_id = 'VIDEO_ID'

youtube = build('youtube', 'v3', developerKey=api_key)

comments = []
next_page_token = None

while True:
    response = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        maxResults=100,
        pageToken=next_page_token,
        textFormat='plainText'
    ).execute()

    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append({
            'author': comment['authorDisplayName'],
            'text': comment['textDisplay'],
            'published_at': comment['publishedAt'],
            'like_count': comment['likeCount']
        })

    next_page_token = response.get('nextPageToken')
    if not next_page_token:
        break

df = pd.DataFrame(comments)
df.to_csv('youtube_comments.csv', index=False)
print("Selesai. Komentar disimpan.")
