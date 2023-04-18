
def 中国平安年报查询(input_text):
    pinecone.init(api_key="bd20d2c3-f100-4d24-954b-c17928d1c2da",  # find at app.pinecone.io
                      environment="us-east4-gcp",  # next to api key in console
                      namespace="ZGPA_601318")
    index = pinecone.Index(index_name="kedu")
    a=embeddings.embed_query(input_text)
    www=index.query(vector=a, top_k=1, namespace='ZGPA_601318', include_metadata=True)
    c = [x["metadata"]["text"] for x in www["matches"]]
    return c
def 双汇发展年报查询(input_text):
    namespace="ShHFZ_000895"
    pinecone.init(api_key="bd20d2c3-f100-4d24-954b-c17928d1c2da",  # find at app.pinecone.io
                      environment="us-east4-gcp",  # next to api key in console
                      namespace=namespace)
    index = pinecone.Index(index_name="kedu")
    a=embeddings.embed_query(input_text)
    www=index.query(vector=a, top_k=1, namespace=namespace, include_metadata=True)
    c = [x["metadata"]["text"] for x in www["matches"]]
    return c
