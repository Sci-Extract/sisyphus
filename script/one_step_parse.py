import glob
import os
import shutil
import argparse
import subprocess

def collect_articles_from_crawler(where, to):
    os.makedirs(to, exist_ok=True)
    for publisher in os.listdir(where):
        for article_dir in os.listdir(os.path.join(where, publisher)):
            article_dir_path = os.path.join(where, publisher, article_dir) 
            file_paths = glob.glob(os.path.join(article_dir_path, '*.html'))
            for file_path in file_paths:
                file_name = file_path.split(os.sep)[-1]
                shutil.copy(file_path, os.path.join(to, file_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='transfer crawler downloaded articles to a single dir')
    parser.add_argument(
        '--where',
        help='where the downloaded articles saved',
        default='data_articles'
    )
    parser.add_argument(
        '--to',
        help='move to which dir',
        default='articles_unprocessed'
    )
    
    args = parser.parse_args()
    collect_articles_from_crawler(args.where, args.to)
    script_loc = os.path.join('script', 'process_articles.py')
    subprocess.run(['python', script_loc, '--input_dir', args.to], check=True)
