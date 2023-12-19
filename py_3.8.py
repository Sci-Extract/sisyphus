"""File structure should align with standard format, will parse .xml or .html file to plain text file with utf-8 encoding."""

import argparse
import os

from chemdataextractor import Document
from chemdataextractor.doc import Paragraph

def parse(in_file, out_file):
    with open(in_file, "rb") as fo:
        doc = Document.from_file(fo)
        
    paras: Paragraph = doc.paragraphs
    paras_raw_text = [str(para) for para in paras]

    with open(out_file, "+w", encoding='utf-8') as f:
        for para in paras_raw_text:
            # f.write(para)
            strip_newline_para = para.replace('\n', ' ')
            f.write(strip_newline_para)

def get_target_dir_html_xml(dir_path: str) -> str:
    files = os.listdir(dir_path)
    html_xml = [file for file in files if file.endswith('.html') or file.endswith('.xml')][0] # normally there is one file with extension xml/html
    file_path = os.path.join(dir_path, html_xml)
    return file_path

def batch_conversion(source):
    for publisher in os.listdir(source):
        publisher_dir = os.path.join(source, publisher)
        for article in os.listdir(publisher_dir):
            article_path = os.path.join(publisher_dir, article)
            file_path = get_target_dir_html_xml(article_path)
            if file_path:
                name_without_extension = os.path.splitext(os.path.basename(file_path))[0]
                out_path = os.path.join(article_path, name_without_extension + '.txt')
                parse(file_path, out_path)
            else:
                continue

batch_conversion("data_articles")
    