from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def ingest_docs():
    loader = ReadTheDocsLoader("langchain-docs/api.python.langchain.com/en/latest", encoding='utf-8')

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        # print("new_url",new_url)
        new_url = new_url.replace("langchain-docs", "https:/").replace("\\","/")
        # print("updated new url: ", new_url)
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    BATCH_SIZE = 250
    for i in range(0, len(documents),BATCH_SIZE):
        doc_ = documents[i:i+BATCH_SIZE]
        print("documents reference",i,i+BATCH_SIZE)
        PineconeVectorStore.from_documents(
            doc_, embeddings, index_name="langchain-doc-index"
        )
    print("****Loading to vectorstore done ***")
    # for doc in documents[0:10]:
    #     print("\n************",doc)


if __name__ == "__main__":
    ingest_docs()
