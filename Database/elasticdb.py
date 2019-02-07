import elasticsearch
from datetime import datetime
from elasticsearch import Elasticsearch
es = Elasticsearch()

doc = {
    'creator':'omar',
    'text':'kjsahdkjsahdsakjdhsk',
    'timestamp': datetime.now()
}

res = es.index(index='test-index', doc_type='post', id=3, body=doc)

print(res['result'])

res = es.get(index='test-index', doc_type='post', id=1)
print(res['_source'])

es.indices.refresh(index='test-index')

res = es.search(index='test-index', body={'query': {'match_all': {}}})
print(res['hits']['total'])
for hit in res['hits']['hits']:
    print(hit['_source'])
''''''