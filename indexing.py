# -*- coding:utf-8 -*-
"""
@File    :   test_aindex.py
@Time    :   2024/05/05 18:30:34
@Author  :   soike
@Version :   1.0
@Contact :   luvusoike@icloud.com
@License :   MIT Lisence
@Desc    :   None
"""

import argparse

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
    args = parser.parse_args()
    acreate_vectordb(file_folder=args.directory, collection_name=args.collection)

if __name__ == '__main__':
    main()
