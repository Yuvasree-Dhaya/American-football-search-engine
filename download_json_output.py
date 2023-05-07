import requests
import json

url = "http://localhost:8983/solr/nutch/select?q=*:*&wt=json&rows=10"
response = requests.get(url)
data = response.json()

with open("mydata.json", "w") as f:
    json.dump(data, f)