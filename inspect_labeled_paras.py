from script.file_name_doi_conversion import doi_to_file_name
from sisyphus.utils.helper_functions import get_plain_articledb
from sisyphus.strategy.utils import get_paras_with_props
from sisyphus.chain import Paragraph

def load(docs):
    return [Paragraph.from_labeled_document(doc, id_) for id_, doc in enumerate(docs)]

if __name__ == '__main__':
    import sys
    doi = sys.argv[1]

db = 'labeled_'
articledb = get_plain_articledb(db)

paras_iter = articledb.get(doi_to_file_name(doi))
paras = load(list(paras_iter) if paras_iter is not None else [])
properties = ['synthesis', 'phase',  'grain_size', 'strength']


def get_text(para):
    if hasattr(para, 'page_content'):
        return para.page_content or ''
    if isinstance(para, dict):
        return para.get('page_content') or para.get('text') or ''
    return str(para)

paras_text = [get_text(p) for p in paras]

prop_indices = {}
for prop in properties:
    if prop == 'synthesis':
        subset = [p for p in paras if getattr(p, 'is_synthesis', False)]
    else:
        subset = get_paras_with_props(paras, prop) or []
    idxs = set()
    for sp in subset:
        st = get_text(sp)
        try:
            idxs.add(paras_text.index(st))
        except ValueError:
            # fallback: compare by identity
            for i, original in enumerate(paras):
                if original is sp:
                    idxs.add(i)
                    break
    prop_indices[prop] = idxs

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")

html = []
html.append('<!doctype html><html><head><meta charset="utf-8"><title>Annotated paragraphs</title>')
html.append('<style>'
            'body{font-family:Arial,Helvetica,sans-serif;padding:20px}'
            '.para{border:1px solid #ddd;padding:10px;margin:10px 0;border-radius:6px}'
            '.tags{margin-top:8px}'
            '.tag{display:inline-block;background:#eee;color:#222;padding:4px 8px;border-radius:4px;margin-right:6px;font-size:12px}'
            '.tag.strength{background:#ffdede}'
            '.tag.phase{background:#dedefd}'
            '.tag.grain_size{background:#deffd8}'
            '.tag.synthesis{background:#fff4ce}'
            '.section{margin-bottom:20px}'
            '</style></head><body>')
html.append('<h1>Annotated paragraphs (grouped by property)</h1>')
html.append('<div class="legend"><strong>Legend:</strong> ' + ' '.join(f'<span class="tag {p}">{p}</span>' for p in properties) + '</div>')

# For each property, show only paragraphs that have that property.
# Paragraphs with multiple properties will appear in multiple property sections.
for prop in properties:
    indices = sorted(prop_indices.get(prop, []))
    if not indices:
        continue  # skip empty sections
    html.append(f'<div class="section"><h2>{prop} ({len(indices)})</h2>')
    for i in indices:
        text = paras_text[i]
        # Optionally show which other properties this paragraph has
        other_tags = ''.join(f'<span class="tag {p}">{p}</span>' for p in properties if i in prop_indices.get(p, ()) )
        html.append(f'<div class="para"><div><strong>Paragraph {i+1}</strong></div>'
                    f'<div>{esc(text)}</div>'
                    f'<div class="tags">{other_tags}</div></div>')
    html.append('</div>')

html.append('</body></html>')

out_path = 'labeled_paragraphs.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write("\n".join(html))

print(f'Wrote {out_path}')
