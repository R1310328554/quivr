from elasticsearch import Elasticsearch

# elastic J7ptEAnl60OahmyAHWeGYqYW
# credentials-67be05-2023-Nov-27-

# 实例化一个ip为localhost，端口为9200，允许超时一小时的es对象
# es = Elasticsearch(hosts="localhost",port=9200,timeout=3600)

# # 查询test索引前10条数据
# print(es.search(index='test'))


from datetime import datetime
from elasticsearch import Elasticsearch
# es = Elasticsearch()

es = Elasticsearch(
    cloud_id="My_deployment:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDI1YWE4ODE5ZDI2ODRiOWM4YTdhYzJhNTBjMjRiYzhhJDUxMzc0Y2I2YzcxODRkMGJiYzA3NDlhMGJhMjI2NTdj",
#    cloud_id="cluster-1:dXMa5Fx...",
    http_auth=("elastic", "J7ptEAnl60OahmyAHWeGYqYW"),
)

doc = {
    'author': 'kimchy',
    'text': 'Elasticsearch: cool. bonsai cool.',
    'timestamp': datetime.now(),
}
res = es.index(index="test-index", id=1, body=doc)
print(res['result'])

res = es.get(index="test-index", id=1)
print(res['_source'])

es.indices.refresh(index="test-index")

res = es.search(index="test-index", body={"query": {"match_all": {}}})
print("Got %d Hits:" % res['hits']['total']['value'])
for hit in res['hits']['hits']:
    print("%(timestamp)s %(author)s: %(text)s" % hit["_source"])