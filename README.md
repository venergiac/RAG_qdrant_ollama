# Using OLLAMA with vector db QDRANT for the best privacy of your data - implementing as simple RAG

You can find here a complete python code to build a simple RAG with OLLAMA and QADRANT.

## Prerequisites:
* OLLAMA installed
* Docker installed
* Python insalled

## Prepare env
From bash run

    pip install ollama qdrant-client pandas
    docker run -p 6333:6333 qdrant/qdrant
    ollama pull qwen3-embedding

## Test the cliend and ollama
Now you can test Ollama and Qadrant


    from qdrant_client import QdrantClient, models
    import ollama
    
    COLLECTION_NAME = "NicheApplications"
    
    # Initialize Ollama client
    oclient = ollama.Client(host="localhost")
    
    # Initialize Qdrant client
    qclient = QdrantClient(host="localhost", port=6333)
    
    # Text to embed
    text = "Ollama excels in niche applications with specific embeddings"
    
    # Generate embeddings
    model_name="qwen3-embedding"
    
    response = oclient.embeddings(model=model_name, prompt=text)
    embeddings = response["embedding"]
    
    # Create a collection if it doesn't already exist
    if not qclient.collection_exists(COLLECTION_NAME):
        qclient.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=len(embeddings), distance=models.Distance.COSINE
            ),
        )
    
    # Upload the vectors to the collection along with the original text as payload
    qclient.upsert(
        collection_name=COLLECTION_NAME,
        points=[models.PointStruct(id=1, vector=embeddings, payload={"text": text})],
    )


See jupyter notebook for a simple RAG


## Credits
* (venergiac blog)[https://venergiac.blogspot.com/]
* (QDRANT)[https://qdrant.tech/documentation/embeddings/ollama/]
* (RAG PROMPTING)[https://iamholumeedey007.medium.com/prompt-engineering-patterns-for-successful-rag-implementations-b2707103ab56]




