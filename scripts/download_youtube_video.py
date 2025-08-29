import yt_dlp
import os
import argparse

def download_youtube_video(youtube_url, output_path):
    """
    Downloads a YouTube video to the specified output path.
    """
    ydl_opts = {
        # Explicitly request h264 mp4 format
        'format': 'bestvideo[ext=mp4][vcodec^=avc]+bestaudio[ext=m4a]/mp4',
        'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'),
        'merge_output_format': 'mp4',
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4'
        }],
        'ffmpeg_location': '/usr/bin/ffmpeg' # Ensure ffmpeg is in PATH or specify its location
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            video_filename = ydl.prepare_filename(info_dict)
            print(f"Successfully downloaded: {video_filename}")
            return video_filename
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a YouTube video.")
    parser.add_argument('--url', type=str, required=True, help="YouTube video URL.")
    parser.add_argument('--output_dir', type=str, default="/mnt/d/progress/ani_bender/input_videos", help="Directory to save the downloaded video.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    download_youtube_video(args.url, args.output_dir)