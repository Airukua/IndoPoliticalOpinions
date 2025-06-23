import argparse
import json
import os
from scrapper.scrapper_data import fetch_and_save_comments
from utils.data_loader import DataLoader

def parse_video_map(video_args):
    video_map = {}
    for pair in video_args:
        if '=' not in pair:
            raise argparse.ArgumentTypeError("Video arguments must be in the form VIDEO_ID=FILENAME.csv")
        video_id, filename = pair.split('=', 1)
        video_map[video_id] = filename
    return video_map


def main():
    parser = argparse.ArgumentParser(description="Fetch and anonymize YouTube video comments.")
    parser.add_argument('--api-key', required=True, help='YouTube Data API key')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--videos', nargs='+', help='List of video_id=filename.csv pairs (e.g., abc123=out1.csv xyz456=out2.csv)')
    group.add_argument('--video-map-json', help='Path to JSON file with video_id to filename mapping')

    parser.add_argument('--output-dir', default='scrapper', help='Directory to save the CSV files')

    args = parser.parse_args()

    if args.video_map_json:
        if not os.path.isfile(args.video_map_json):
            raise FileNotFoundError(f"JSON file not found: {args.video_map_json}")
        with open(args.video_map_json, 'r') as f:
            video_map = json.load(f)
    else:
        video_map = parse_video_map(args.videos)

    fetch_and_save_comments(api_key=args.api_key, video_map=video_map, output_dir=args.output_dir)

if __name__ == '__main__':
    main()
