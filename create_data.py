import os
import json
from xml.dom.minidom import Document 


with open('val_dataset.json', encoding='utf-8') as f:
    data = json.load(f)

dataset = {}

dataset['lang'] = 'vi'
dataset['version'] = '1'
dataset['split'] = 'val'

documents = []

count = 1
for i, key in enumerate(data.keys()):
    document = {}
    document['id'] = key.split('.')[0]
    document['uid'] = ''
    
    doc = []
    
    for d in data[key]['cells']:
        if d['cat_id'] == '0':
            label = 'title'
        elif d['cat_id'] == '1':
            label = 'header'
        elif d['cat_id'] == '2' or '3':
            label = 'paragraph'
        elif d['cat_id'] == 5:
            label = 'other'
        else:
            print(d['cat_id'])
            print('alr')
            exit()
        poly = d['poly']
        doc.append({
            'box': [min(poly[0], poly[2], poly[4], poly[6]), min(poly[1], poly[3], poly[5], poly[7]),
                    max(poly[0], poly[2], poly[4], poly[6]), max(poly[1], poly[3], poly[5], poly[7])],
            'text': d['vietocr_text'],
            'label': label,
            'words': [],
            'linking': [],
            'id': count
                    })
        count+=1
    
    document['document'] = doc
    document['img'] = {
        'fname': key,
        'width': data[key]['w_origin'],
        'height': data[key]['h_origin']
    }
    
    documents.append(document)

dataset['documents'] = documents

with open('own_val.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False)
    

    