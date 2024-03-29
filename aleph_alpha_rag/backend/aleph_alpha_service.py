"""The script to initialize the Qdrant db backend with aleph alpha."""

import os
import pathlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import nltk
import numpy as np
from aleph_alpha_client import Client, CompletionRequest, ExplanationRequest, Prompt
from dotenv import load_dotenv
from langchain.docstore.document import Document as LangchainDocument
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFium2Loader,
    TextLoader,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from loguru import logger
from omegaconf import DictConfig
from ultra_simple_config import load_config

from aleph_alpha_rag.utils.utility import generate_prompt
from aleph_alpha_rag.utils.vdb import get_db_connection

if TYPE_CHECKING:
    from langchain_community.vectorstores import Qdrant

nltk.download("punkt")  # This needs to be installed for the tokenizer to work.
load_dotenv()

aleph_alpha_token = os.getenv("ALEPH_ALPHA_API_KEY")
tokenizer = None


def get_tokenizer(aleph_alpha_token: str) -> None:
    """Initialize the tokenizer."""
    global tokenizer
    client = Client(token=aleph_alpha_token)
    tokenizer = client.tokenizer("luminous-base")


def count_tokens(text: str) -> int:
    """Count the number of tokens in the text.

    Args:
    ----
        text (str): The text to count the tokens for.

    Returns:
    -------
        int: Number of tokens.
    """
    tokens = tokenizer.encode(text)
    return len(tokens)


class AlephAlphaService:

    """The Aleph Alpha Service. This class is used to interact with the Aleph Alpha API."""

    @load_config(location="config/main.yml")
    def __init__(self, cfg: DictConfig, collection_name: str, aleph_alpha_token: str) -> None:
        """Initialize the Ollama Service."""
        self.cfg = cfg
        self.collection_name = collection_name
        self.aleph_alpha_token = aleph_alpha_token
        self.vector_db: Qdrant = get_db_connection(collection_name=self.collection_name, aleph_alpha_token=self.aleph_alpha_token)

        if not self.aleph_alpha_token:
            msg = "Token cannot be None or empty."
            raise ValueError(msg)

    def send_completion_request(self, text: str) -> str:
        """Sends a completion request to the Luminous API.

        Args:
        ----
            text (str): The prompt to be sent to the API.
            token (str): The token for the Luminous API.

        Returns:
        -------
            str: The response from the API.

        Raises:
        ------
            ValueError: If the text or token is None or empty, or if the response or completion is empty.
        """
        if not text:
            msg = "Text cannot be None or empty."
            raise ValueError(msg)

        client = Client(token=self.aleph_alpha_token)

        request = CompletionRequest(
            prompt=Prompt.from_text(text),
            maximum_tokens=self.cfg.aleph_alpha_completion.max_tokens,
            stop_sequences=[self.cfg.aleph_alpha_completion.stop_sequences],
            repetition_penalties_include_completion=self.cfg.aleph_alpha_completion.repetition_penalties_include_completion,
        )
        response = client.complete(request, model=self.cfg.aleph_alpha_completion.model)

        # ensure that the response is not empty
        if not response.completions:
            msg = "Response is empty."
            raise ValueError(msg)

        # ensure that the completion is not empty
        if not response.completions[0].completion:
            msg = "Completion is empty."
            raise ValueError(msg)

        return str(response.completions[0].completion)

    def embedd_documents(self, directory: str, file_ending: str = "*.pdf") -> None:
        """Embeds the documents in the given directory in the Aleph Alpha database.

        This method uses the Directory Loader for PDFs and the PyPDFium2Loader to load the documents.
        The documents are then added to the Qdrant DB which embeds them without deleting the old collection.

        Args:
        ----
            directory (str): The directory containing the PDFs to embed.
            aleph_alpha_token (str): The Aleph Alpha API token.
            file_ending (str, optional): The file ending of the files to embed. Defaults to "*.pdf".

        Returns:
        -------
            None
        """
        vector_db: Qdrant = get_db_connection(collection_name=self.collection_name, aleph_alpha_token=self.aleph_alpha_token)

        if file_ending == "*.pdf":
            loader = DirectoryLoader(directory, glob=file_ending, loader_cls=PyPDFium2Loader)
        elif file_ending == "*.txt":
            loader = DirectoryLoader(directory, glob=file_ending, loader_cls=TextLoader)
        else:
            msg = "File ending not supported."
            raise ValueError(msg)

        get_tokenizer(self.aleph_alpha_token)

        splitter = NLTKTextSplitter(length_function=count_tokens, chunk_size=300, chunk_overlap=50)
        docs = loader.load_and_split(splitter)

        logger.info(f"Loaded {len(docs)} documents.")
        text_list = [doc.page_content for doc in docs]
        metadata_list = [doc.metadata for doc in docs]

        for m in metadata_list:
            # only when there are / in the source
            if "/" in m["source"]:
                m["source"] = m["source"].split("/")[-1]

        vector_db.add_texts(texts=text_list, metadatas=metadata_list)

        logger.info("SUCCESS: Texts embedded.")

    def embedd_text_files(self, folder: str, seperator: str) -> None:
        """Embeds text files in the Aleph Alpha database.

        Args:
        ----
            folder (str): The folder containing the text files to embed.
            aleph_alpha_token (str): The Aleph Alpha API token.
            seperator (str): The seperator to use when splitting the text into chunks.

        Returns:
        -------
            None
        """
        vector_db: Qdrant = get_db_connection(collection_name=self.collection_name, aleph_alpha_token=self.aleph_alpha_token)

        # iterate over the files in the folder
        for file in os.listdir(folder):
            # check if the file is a .txt or .md file
            if not file.endswith((".txt", ".md")):
                continue

            # read the text from the file
            text = pathlib.Path(pathlib.Path(folder) / file).read_text()
            text_list: list = text.split(seperator)

            # check if first and last element are empty
            if not text_list[0]:
                text_list.pop(0)
            if not text_list[-1]:
                text_list.pop(-1)

            # ensure that the text is not empty
            if not text_list:
                msg = "Text is empty."
                raise ValueError(msg)

            logger.info(f"Loaded {len(text_list)} documents.")
            # get the name of the file
            file_path = Path(file)
            metadata = file_path.stem
            # add _ and an incrementing number to the metadata
            metadata_list: list = [{"source": f"{metadata}_{i!s}", "page": 0} for i in range(len(text_list))]
            vector_db.add_texts(texts=text_list, metadatas=metadata_list)

        logger.info("SUCCESS: Text embedded.")

    def search_documents_aleph_alpha(self, query: str, amount: int = 1, threshold: float = 0.0) -> list[tuple[LangchainDocument, float]]:
        """Searches the Aleph Alpha service for similar documents.

        Args:
        ----
            aleph_alpha_token (str): Aleph Alpha API Token.
            query (str): The query that should be searched for.
            amount (int, optional): The number of documents to return. Defaults to 1.
            threshold (float, optional): The threshold for the similarity score. Defaults to 0.0.

        Returns:
        -------
            List[Tuple[Document, float]]: A list of tuples containing the documents and their similarity scores.
        """
        if not query:
            msg = "Query cannot be None or empty."
            raise ValueError(msg)
        if amount < 1:
            msg = "Amount must be greater than 0."
            raise ValueError(msg)
        # TODO: FILTER
        try:
            vector_db: Qdrant = get_db_connection(collection_name=self.collection_name, aleph_alpha_token=self.aleph_alpha_token)
            docs = vector_db.similarity_search_with_score(query=query, k=amount, score_threshold=threshold)
            logger.info("SUCCESS: Documents found.")
        except ValueError as e:
            msg = f"Failed to search documents: {e}"
            logger.error(f"ERROR:{msg}: {e}")
            raise ValueError(msg) from e
        return docs

    def qa(
        self,
        documents: list[tuple[LangchainDocument, float]],
        query: str,
        summarization: bool = False,
    ) -> tuple[str, str, dict[Any, Any] | list[dict[Any, Any]]]:
        """QA takes a list of documents and returns a list of answers.

        Args:
        ----
            aleph_alpha_token (str): The Aleph Alpha API token.
            documents (List[Tuple[Document, float]]): A list of tuples containing the document and its relevance score.
            query (str): The query to ask.
            summarization (bool, optional): Whether to use summarization. Defaults to False.

        Returns:
        -------
            Tuple[str, str, Union[Dict[Any, Any], List[Dict[Any, Any]]]]: A tuple containing the answer, the prompt, and the metadata for the documents.
        """
        # TODO: improve this code
        # if the list of documents contains only one document extract the text directly
        if len(documents) == 1:
            text = documents[0][0].page_content
            meta_data = documents[0][0].metadata
        else:
            # extract the text from the documents
            texts = [doc[0].page_content for doc in documents]
            text = "".join(self.summarize_text_aleph_alpha(t) for t in texts) if summarization else " ".join(texts)
            meta_data = [doc[0].metadata for doc in documents]

        # load the prompt
        prompt = generate_prompt("qa.j2", text=text, query=query)

        try:
            # call the luminous api
            answer = self.send_completion_request(prompt)

        except ValueError as e:
            # if the code is PROMPT_TOO_LONG, split it into chunks
            if e.args[0] == "PROMPT_TOO_LONG":
                logger.info("Prompt too long. Summarizing.")

                # summarize the text
                short_text = self.summarize_text_aleph_alpha(text)

                # generate the prompt
                prompt = generate_prompt("qa.j2", text=short_text, query=query)

                # call the luminous api
                answer = self.send_completion_request(prompt)

        # extract the answer
        return answer, prompt, meta_data

    def explain_qa(self, document: LangchainDocument, query: str) -> tuple[str, float, str, str, dict[Any, Any] | list[dict[Any, Any]]]:
        """Explain QA."""
        text = document[0][0].page_content
        meta_data = document[0][0].metadata

        # load the prompt
        prompt = generate_prompt("qa.j2", text=text, query=query)

        answer = self.send_completion_request(text=prompt)

        exp_req = ExplanationRequest(
            Prompt.from_text(prompt),
            answer,
            control_factor=0.1,
            prompt_granularity="sentence",
            normalize=True,
        )
        client = Client(token=self.aleph_alpha_token)

        response_explain = client.explain(exp_req, model=self.cfg.aleph_alpha_completion.model)
        explanations = response_explain.explanations[0].items[0].scores

        threshold = 0.7
        # if all of the scores are belo 0.7 raise an error
        if all(item.score < threshold for item in explanations):
            msg = f"All scores are below {threshold}."
            raise ValueError(msg)

        # remove element if the text contains Response: or Instructions:
        for exp in explanations:
            txt = prompt[exp.start : exp.start + exp.length]
            if "Response:" in txt or "Instruction:" in txt:
                explanations.remove(exp)

        # pick the top explanation based on score
        top_explanation = max(explanations, key=lambda x: x.score)

        # get the start and end of the explanation
        start = top_explanation.start
        end = top_explanation.start + top_explanation.length

        # get the explanation from the prompt
        explanation = prompt[start:end]

        # get the score
        score = np.round(top_explanation.score, decimals=3)

        # get the text from the document
        text = document[0][0].page_content

        return explanation, score, text, answer, meta_data

    def qa_chain(self, query: str) -> None:
        """QA Chain Imp."""
        from langchain.llms import AlephAlpha

        model = AlephAlpha(
            model="luminous-extended",
            maximum_tokens=20,
            stop_sequences=["Q:"],
            aleph_alpha_api_key=self.aleph_alpha_token,
        )
        vector_db: Qdrant = get_db_connection(collection_name=self.collection_name, aleph_alpha_token=self.aleph_alpha_token)

        retriever = vector_db.as_retriever()

        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model | StrOutputParser()

        answer = chain.invoke(query)

        logger.info(answer)


if __name__ == "__main__":
    token = os.getenv("ALEPH_ALPHA_API_KEY")

    if not token:
        msg = "Token cannot be None or empty."
        raise ValueError(msg)

    aa_service = AlephAlphaService(collection_name="aleph_alpha", aleph_alpha_token=token)

    # aa_service.embedd_documents("tests/resources")

    docs = aa_service.search_documents_aleph_alpha(query="What are Attentions?", amount=3)

    # logger.info(docs)

    answer, prompt, meta_data = aa_service.qa(documents=docs, query="What are Attentions?")
    logger.info(answer)
    explanation, score, text, answer, meta_data = aa_service.explain_qa(document=docs, query="What are Attentions?")
    logger.info(explanation)
