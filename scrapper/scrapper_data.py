# ------------------------------------------------------------------------------
# MIT License
# Copyright (c) 2025 Abdul Wahid Rukua
#
# This code is open-source under the MIT License.
# See LICENSE file in the root of the repository for full license information.
# ------------------------------------------------------------------------------

from googleapiclient.discovery import build
import pandas as pd
from typing import Dict, List, Optional
import os
from scrapper.anonymization import anonymize_authors_globally

def fetch_youtube_comments(api_key: str, video_id: str) -> List[Dict[str, Optional[str]]]:
    """
    Fetches all top-level comments from a YouTube video using the YouTube Data API.

    :param api_key: Your YouTube Data API key.
    :param video_id: The YouTube video ID.
    :return: A list of comment dictionaries containing author, text, timestamp, and like count.
    """
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

        for item in response.get('items', []):
            snippet = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'author': snippet.get('authorDisplayName'),
                'text': snippet.get('textDisplay'),
                'published_at': snippet.get('publishedAt'),
                'like_count': snippet.get('likeCount')
            })

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return comments

def save_comments_to_csv(comments: List[Dict[str, Optional[str]]], output_path: str) -> None:
    """
    Saves a list of comments to a CSV file.

    :param comments: A list of comment dictionaries.
    :param output_path: Destination path for the CSV file.
    """
    df = pd.DataFrame(comments)
    df.to_csv(output_path, index=False)
    print(f"[SAVED] {len(df)} comments → {output_path}")

def fetch_and_save_comments(api_key: str, video_map: Dict[str, str], output_dir: str = "scrapper") -> None:
    """
    Coordinates fetching and saving YouTube comments for multiple videos,
    and then applies global anonymization to the 'author' field.

    :param api_key: Your YouTube Data API key.
    :param video_map: A dictionary mapping video IDs to their output CSV filenames.
    :param output_dir: The directory where CSV files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_paths = []

    for video_id, filename in video_map.items():
        print(f"[INFO] Fetching comments for video: {video_id}")
        comments = fetch_youtube_comments(api_key, video_id)
        output_path = os.path.join(output_dir, filename)
        save_comments_to_csv(comments, output_path)
        file_paths.append(output_path)

    print("[INFO] Applying global author anonymization...")
    anonymize_authors_globally(file_paths)
    print("[DONE] Anonymization complete.")
