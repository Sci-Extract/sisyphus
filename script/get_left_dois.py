import os
import re
import argparse
import glob

from sisyphus.utils.utilities import read_wos_excel
from sisyphus.crawler.publishers_config import publishers_doi_prefix


target = 'data_articles'
def get_downloaded_doi(target):
    ls = []
    for publisher in os.listdir(target):
        pub_dir = os.path.join(target, publisher)
        for suffix in os.listdir(pub_dir):
            ls.append(publishers_doi_prefix[publisher] + '/' + suffix)
    print(f'you have already downloaded {len(ls)} files')
    return set(ls) 

def get_diff(dowloaded, origin):
    if origin.endswith('.txt'):
        with open(origin, 'r', encoding='utf8') as f:
            doi_ls = f.readlines()
            doi_ls = [doi.strip() for doi in doi_ls]
    elif origin.endswith('.xlsx'):
        doi_ls = read_wos_excel(origin)
    valid_prefix = set(publishers_doi_prefix.values())
    def filter_func(doi):
        match = re.match(r'\d{2}\.\d{4}', doi).group()
        in_ = match in valid_prefix
        return in_
    filtered_doi = set(filter(filter_func, doi_ls))
    return filtered_doi.difference(dowloaded)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'doi_file',
        help='the file contains doi'
    )
    parser.add_argument(
        '--downloaded',
        help='where your files downloaded at',
        default='data_articles'
    )
    args = parser.parse_args()
    
    left_dois = get_diff(get_downloaded_doi(args.downloaded), args.doi_file)
    if not left_dois:
        print('\nalready downloaded all dois')
    else:
        with open('doi_to_download.txt', 'w', encoding='utf8') as f:
            for doi in left_dois:
                f.write(doi + '\n')
        print('\nleft dois wrote to doi_to_download.txt, check it out')
        