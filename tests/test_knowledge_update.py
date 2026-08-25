import unittest

from langchain_core.documents import Document
from scipy.sparse import csr_matrix

from rag_qa.core.document_processor import _split_documents
from rag_qa.core.vector_store import VectorStore


DOCUMENT_ID = "11111111-1111-1111-1111-111111111111"


class FakeMilvusClient:
    def __init__(self):
        self.upserts = []
        self.deletes = []

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def flush(self, **kwargs):
        return None

    def delete(self, **kwargs):
        self.deletes.append(kwargs)
        return {"delete_count": 1}


class KnowledgeUpdateTest(unittest.TestCase):
    def _chunks(self, version):
        document = Document(
            page_content="第一段知识。" * 80,
            metadata={
                "file_path": "demo.txt",
                "file_name": "demo.txt",
                "source": "ai",
                "timestamp": "now",
            },
        )
        return _split_documents(
            [document],
            parent_chunk_size=120,
            child_chunk_size=50,
            chunk_overlap=10,
            document_id=DOCUMENT_ID,
            document_version=version,
            content_hash=f"hash-v{version}",
        )

    def test_chunk_ids_stay_stable_between_versions(self):
        chunks_v1 = self._chunks(1)
        chunks_v2 = self._chunks(2)

        self.assertGreater(len(chunks_v1), 0)
        self.assertEqual(chunks_v1[0].metadata["id"], chunks_v2[0].metadata["id"])
        self.assertEqual(chunks_v2[0].metadata["document_version"], 2)

    def test_vector_metadata_and_version_cleanup_filter(self):
        chunks = self._chunks(2)
        store = VectorStore.__new__(VectorStore)
        store.collection_name = "test"
        store.client = FakeMilvusClient()
        store.embedding_function = lambda texts: {
            "dense": [[0.1, 0.2] for _ in texts],
            "sparse": csr_matrix(
                [[0.0, 0.5, 0.0, 0.2] for _ in texts]
            ),
        }

        count = store.add_documents(chunks[:2])
        written = store.client.upserts[0]["data"][0]
        self.assertEqual(count, 2)
        self.assertEqual(written["id"], chunks[0].metadata["id"])
        self.assertEqual(written["document_id"], DOCUMENT_ID)
        self.assertEqual(written["document_version"], 2)

        store.delete_document_versions_before(DOCUMENT_ID, 2)
        store.delete_by_document_id(DOCUMENT_ID)
        self.assertTrue(
            store.client.deletes[0]["filter"].endswith("document_version < 2")
        )
        self.assertEqual(
            store.client.deletes[1]["filter"],
            f'document_id == "{DOCUMENT_ID}"',
        )


if __name__ == "__main__":
    unittest.main()
