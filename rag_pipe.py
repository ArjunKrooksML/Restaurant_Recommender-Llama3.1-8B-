from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import config

def get_llm(model_n=config.OLLAMA_MODEL):
    return Ollama(model=model_n, base_url=config.OLLAMA_HOST_URL)

def format_docs(docs) -> str:
    formatted_context = "\n\n---\n\n".join(
        f"Restaurant: {doc.metadata.get('name', 'N/A')}\n"
        f"Location: {doc.metadata.get('location', 'N/A')}, {doc.metadata.get('city', 'N/A')}\n"
        f"Cuisines: {doc.metadata.get('cuisines', 'N/A')}\n"
        f"Rating: {doc.metadata.get('rate', 'N/A')}\n"
        f"Cost for Two: {doc.metadata.get('cost_for_two', 'N/A')}\n"
        f"Details: {doc.page_content}"
        for doc in docs
    )
    return formatted_context

PROMPT_TEMPLATE = """
You are an intelligent and helpful assistant for a local restaurant recommendation app.

Using the retrieved information below (labeled with restaurant names), answer the user's question based strictly on the available context.
- If the answer is not present in the context, say: "Sorry, this information is not available in the current restaurant data."
- Do not guess or add any information that is not supported by the context.
- Be clear, concise, and user-friendly.
- If suggesting multiple restaurants, format them in a readable bullet list with key details (name, location, cuisine, etc.).
- **After providing the main answer, list the names of the primary Restaurants from the 'RESTAURANT DATA' section below that you used to formulate your answer, under a heading like 'Sources:'.**

RESTAURANT DATA:
{context}

USER QUESTION:
{question}

RECOMMENDED RESPONSE:
"""

def rag_seq(ret, llmi):
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    ragdocs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llmi
        | StrOutputParser()
    )
    rag_chain_with_source = RunnableParallel(
        {"context": ret, "question": RunnablePassthrough()}
    ).assign(answer=ragdocs)

    return rag_chain_with_source

def get_answer_from_chain(query_str: str, chain):
    return chain.invoke(query_str)
