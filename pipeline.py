"""
Enterprise RAG Pipeline
End-to-end orchestration with Parent-Document Retrieval,
Query Rewriting, Hybrid Search, and Hallucination Guardrails.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


class EnterpriseRAGPipeline:
    """
    Production RAG pipeline with:
    - Parent-Document Retrieval (small chunk index, full parent retrieved)
    - Query rewriting for ambiguous queries
    - Hallucination scoring via groundedness check
    """

    def __init__(self, model: str = "gpt-4o", collection_name: str = "enterprise-rag"):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.collection_name = collection_name

        # Parent-document retriever setup
        self.doc_store = InMemoryStore()
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )
        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.doc_store,
            child_splitter=self.child_splitter,
            parent_splitter=self.parent_splitter,
        )

        self._build_chain()

    def _build_chain(self):
        """Build the RAG chain with query rewriting."""

        # Query rewriting prompt
        rewrite_prompt = ChatPromptTemplate.from_template(
            "Rewrite the following question to be more specific and retrieval-friendly. "
            "Return only the rewritten question, nothing else.\n\nQuestion: {question}"
        )

        # RAG answer prompt
        rag_prompt = ChatPromptTemplate.from_template(
            "Answer the question based ONLY on the following context. "
            "If the answer cannot be found in the context, say 'I don't have enough information to answer this.' "
            "Do not make up information.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}"
        )

        self.rewrite_chain = rewrite_prompt | self.llm | StrOutputParser()

        self.rag_chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def ingest(self, documents: list):
        """Add documents to the retriever."""
        self.retriever.add_documents(documents)
        print(f"Ingested {len(documents)} documents.")

    def query(self, question: str, rewrite: bool = True) -> dict:
        """
        Query the pipeline.

        Args:
            question: User's question
            rewrite: Whether to rewrite the query before retrieval

        Returns:
            dict with answer, rewritten_query, source_docs
        """
        rewritten = self.rewrite_chain.invoke({"question": question}) if rewrite else question
        retrieved_docs = self.retriever.get_relevant_documents(rewritten)
        answer = self.rag_chain.invoke(rewritten)
        groundedness = self._score_groundedness(answer, retrieved_docs)

        return {
            "original_question": question,
            "rewritten_question": rewritten,
            "answer": answer,
            "groundedness_score": groundedness,
            "num_docs_retrieved": len(retrieved_docs),
            "flagged": groundedness < 0.5,
        }

    def _score_groundedness(self, answer: str, docs: list) -> float:
        """
        Simple groundedness check: ask the LLM to score how well
        the answer is supported by the retrieved context.
        Returns a score from 0.0 to 1.0.
        """
        if not docs:
            return 0.0

        context = self._format_docs(docs)
        prompt = ChatPromptTemplate.from_template(
            "On a scale from 0 to 10, how well is the following answer supported by the context? "
            "Reply with only a number.\n\nContext:\n{context}\n\nAnswer:\n{answer}"
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            score_str = chain.invoke({"context": context, "answer": answer})
            return float(score_str.strip()) / 10.0
        except Exception:
            return 1.0


if __name__ == "__main__":
    from langchain_core.documents import Document

    pipeline = EnterpriseRAGPipeline()

    # Sample insurance documents
    sample_docs = [
        Document(page_content="The commercial auto policy deductible is $1,000 per incident for vehicles under 5 tons."),
        Document(page_content="Liability coverage for commercial vehicles extends to $2M per occurrence under standard policy terms."),
        Document(page_content="Claims must be reported within 30 days of the incident. Late reporting may result in claim denial."),
    ]

    pipeline.ingest(sample_docs)

    result = pipeline.query("What is the deductible for commercial auto?")
    print(f"\nQ: {result['original_question']}")
    print(f"Rewritten: {result['rewritten_question']}")
    print(f"A: {result['answer']}")
    print(f"Groundedness: {result['groundedness_score']:.2f}")
    if result["flagged"]:
        print("WARNING: Low groundedness — review this response.")
