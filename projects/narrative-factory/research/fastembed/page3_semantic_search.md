[](https://qdrant.tech/documentation/fastembed/fastembed-semantic-search/#using-fastembed-with-qdrant-for-vector-search)Using FastEmbed with Qdrant for Vector Search
=====================================================================================================================================================================

[](https://qdrant.tech/documentation/fastembed/fastembed-semantic-search/#install-qdrant-client-and-fastembed)Install Qdrant Client and FastEmbed
-------------------------------------------------------------------------------------------------------------------------------------------------

```python
pip install "qdrant-client[fastembed]>=1.14.2"
```

[](https://qdrant.tech/documentation/fastembed/fastembed-semantic-search/#initialize-the-client)Initialize the client
---------------------------------------------------------------------------------------------------------------------

Qdrant Client has a simple in-memory mode that lets you try semantic search locally.

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(":memory:")  # Qdrant is running from RAM.
```

[](https://qdrant.tech/documentation/fastembed/fastembed-semantic-search/#add-data)Add data
-------------------------------------------------------------------------------------------

Now you can add two sample documents, their associated metadata, and a point `id` for each.

```python
docs = [
    "Qdrant has a LangChain integration for chatbots.",
    "Qdrant has a LlamaIndex integration for agents.",
]
metadata = [
    {"source": "langchain-docs"},
    {"source": "llamaindex-docs"},
]
ids = [42, 2]
```

[](https://qdrant.tech/documentation/fastembed/fastembed-semantic-search/#create-a-collection)Create a collection
-----------------------------------------------------------------------------------------------------------------

Qdrant stores vectors and associated metadata in collections. Collection requires vector parameters to be set during creation. In this tutorial, we’ll be using `BAAI/bge-small-en` to compute embeddings.

```python
model_name = "BAAI/bge-small-en"
client.create_collection(
    collection_name="test_collection",
    vectors_config=models.VectorParams(
        size=client.get_embedding_size(model_name), 
        distance=models.Distance.COSINE
    ),  # size and distance are model dependent
)
```

[](https://qdrant.tech/documentation/fastembed/fastembed-semantic-search/#upsert-documents-to-the-collection)Upsert documents to the collection
-----------------------------------------------------------------------------------------------------------------------------------------------

Qdrant client can do inference implicitly within its methods via FastEmbed integration. It requires wrapping your data in models, like `models.Document` (or `models.Image` if you’re working with images)

```python
metadata_with_docs = [
    {"document": doc, "source": meta["source"]} for doc, meta in zip(docs, metadata)
]
client.upload_collection(
    collection_name="test_collection",
    vectors=[models.Document(text=doc, model=model_name) for doc in docs],
    payload=metadata_with_docs,
    ids=ids,
)
```

[](https://qdrant.tech/documentation/fastembed/fastembed-semantic-search/#run-vector-search)Run vector search
-------------------------------------------------------------------------------------------------------------

Here, you will ask a dummy question that will allow you to retrieve a semantically relevant result.

```python
search_result = client.query_points(
    collection_name="test_collection",
    query=models.Document(
        text="Which integration is best for agents?", 
        model=model_name
    )
).points
print(search_result)
```

The semantic search engine will retrieve the most similar result in order of relevance. In this case, the second statement about LlamaIndex is more relevant.

```python
[
    ScoredPoint(
        id=2, 
        score=0.87491801319731,
        payload={
            "document": "Qdrant has a LlamaIndex integration for agents.",
            "source": "llamaindex-docs",
        },
        ...
    ),
    ScoredPoint(
        id=42,
        score=0.8351846627714035,
        payload={
            "document": "Qdrant has a LangChain integration for chatbots.",
            "source": "langchain-docs",
        },
        ...
    ),
]
```

##### Was this page useful?

![Image 3: Thumb up icon](https://qdrant.tech/icons/outline/thumb-up.svg) Yes ![Image 4: Thumb down icon](https://qdrant.tech/icons/outline/thumb-down.svg) No

Thank you for your feedback! 🙏

We are sorry to hear that. 😔 You can [edit](https://qdrant.tech/github.com/qdrant/landing_page/tree/master/qdrant-landing/content/documentation/fastembed/fastembed-semantic-search.md) this page on GitHub, or [create](https://github.com/qdrant/landing_page/issues/new/choose) a GitHub issue.

On this page:

*   [Using FastEmbed with Qdrant for Vector Search](https://qdrant.tech/documentation/fastembed/fastembed-semantic-search/#using-fastembed-with-qdrant-for-vector-search)
    *   [Install Qdrant Client and FastEmbed](https://qdrant.tech/documentation/fastembed/fastembed-semantic-search/#install-qdrant-client-and-fastembed)
    *   [Initialize the client](https://qdrant.tech/documentation/fastembed/fastembed-semantic-search/#initialize-the-client)
    *   [Add data](https://qdrant.tech/documentation/fastembed/fastembed-semantic-search/#add-data)
    *   [Create a collection](https://qdrant.tech/documentation/fastembed/fastembed-semantic-search/#create-a-collection)
    *   [Upsert documents to the collection](https://qdrant.tech/documentation/fastembed/fastembed-semantic-search/#upsert-documents-to-the-collection)
    *   [Run vector search](https://qdrant.tech/documentation/fastembed/fastembed-semantic-search/#run-vector-search)

*   [Edit on Github](https://github.com/qdrant/landing_page/tree/master/qdrant-landing/content/documentation/fastembed/fastembed-semantic-search.md)
*   [Create an issue](https://github.com/qdrant/landing_page/issues/new/choose)

#### Ready to get started with Qdrant?

[Start Free](https://qdrant.to/cloud/)

© 2025 Qdrant.

[Terms](https://qdrant.tech/legal/terms_and_conditions/)[Privacy Policy](https://qdrant.tech/legal/privacy-policy/)[Impressum](https://qdrant.tech/legal/impressum/)

×

[Powered by](https://qdrant.tech/)

About cookies on this site
--------------------------

We use cookies to collect and analyze information on site performance and usage, to provide social media features, and to enhance and customize content and advertisements. [Learn more](https://qdrant.tech/legal/privacy-policy/#cookies-and-web-beacons)

