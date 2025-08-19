from pytube import YouTube 
import os
import argparse


input_file = 'WTour.txt'


def main():
    parser = argparse.ArgumentParser(description='Download WTour videos from WTour.txt')
    parser.add_argument('--output_folder', help='Path to the store videos')
    args = parser.parse_args()

    
    output_base_folder = args.output_folder

    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)

    # exit()
    with open(input_file, 'r') as file:
        for line in file:
            try:
                youtube_link, city = map(str.strip, line.split(','))
                output_folder = os.path.join(output_base_folder, city)

                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)


                #yt = YouTube(youtube_link, use_oauth=True, allow_oauth_cache=True)
                yt=YouTube(youtube_link)
                # downloads the highest res WTour videos. 
                video = yt.streams.filter(adaptive=True, file_extension='mp4').order_by('resolution').desc().first()
                video.download(output_folder)
                print(f"Video downloaded for {city}")
            except Exception as e:
                print(f"Error processing line: {line}\nError: {e}")


    print("All videos downloaded successfully!")


if __name__ == "__main__":
    main()

