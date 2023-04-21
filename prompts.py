# flake8: noqa
from langchain.prompts.prompt import PromptTemplate

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a 
standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

prompt_template = """As a patent practitioner, such as a patent analyst, the content you are currently providing is 
about patent specifications and patent requirements. Use the following context to answer the final question. If you 
don't know the answer, say you don't know. Don't try to make up the answer. Please use questioning language to answer, 
and the answer needs to be more detailed. If the last sentence is not over, there is no need to return.

{context}

Question: {question}
Helpful Answer:"""
QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
