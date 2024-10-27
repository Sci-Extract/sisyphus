"""creating an article database without embedding vectors, just plain text"""
import argparse
from sisyphus.index import create_plaindb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d','--directory',
        help='processed html articles dir',
        default='articles_processed'
    )
    parser.add_argument(
        '--db_name',
        help='the name of the indexed database'
    )
    parser.add_argument(
        '--full_text',
        help='set 0 to disable, set 1 to enable'
    )
    args = parser.parse_args()
    full_text = bool(int(args.full_text))
    create_plaindb(file_folder=args.directory, db_name=args.db_name, full_text=full_text)
