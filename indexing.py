# -*- coding:utf-8 -*-
"""
@File    :   indexing.py
@Time    :   2024/05/05 18:30:34
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   None
"""

import argparse
import logging

from sisyphus.index import acreate_vectordb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        help='the folder which contains cpp parser paresed html files',
        required=True
    )
    parser.add_argument(
        '-c',
        '--collection',
        help='the name of the collection gonna create',
        required=True
    )
    parser.add_argument(
        '-v',
        '--verbose',
        help='set verbose to 1 to enable logging, otherwise showing task bar, default to 0',
        default=0
    )
    parser.add_argument(
        '--local',
        help='set local to 1 to enable local storage of chroma database, default to 0',
        default=0
    )
    args = parser.parse_args()
    verbose = bool(int(args.verbose))
    if verbose:
        logging.basicConfig(level=20)
    else:
        logging.basicConfig(level=30)
    local = bool(int(args.local))
    acreate_vectordb(file_folder=args.directory, collection_name=args.collection, local=local)

if __name__ == '__main__':
    main()
