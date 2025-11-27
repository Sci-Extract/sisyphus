import re

CHAR_TO_HTML_LBS = {
    '/': '&sol;',
    '\\': '&bs;',
    '?': '&qm;',
    '*': '&st;',
    ':': '&cl;',
    '|': '&vb;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    '\'': '&apos;'
}
def file_name_to_doi(file_name):
    reverse_map = {v: k for k, v in CHAR_TO_HTML_LBS.items()}
    pattern = re.compile('|'.join(re.escape(e) for e in sorted(reverse_map.keys(), key=len, reverse=True)))
    doi = pattern.sub(lambda m: reverse_map[m.group(0)], file_name)
    if doi.endswith('.html'):
        doi = doi[:-5]
    return doi

def doi_to_file_name(doi):
    pattern = re.compile('|'.join(re.escape(k) for k in sorted(CHAR_TO_HTML_LBS.keys(), key=len, reverse=True)))
    file_name = pattern.sub(lambda m: CHAR_TO_HTML_LBS[m.group(0)], doi)
    file_name += '.html'
    return file_name
    
if __name__ == '__main__':
    import sys
    arg = sys.argv[1]
    print(doi_to_file_name(arg))
