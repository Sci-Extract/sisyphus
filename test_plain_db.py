"""creating a article database without embedding"""
from sisyphus.index import create_plaindb

create_plaindb(file_folder='your folder', db_name='your db name, end with .db, e.g., plain.db')