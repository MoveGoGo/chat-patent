from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

# os.environ["OPENAI_API_KEY"] = "{your-api-key}"

global retriever


def load_embedding():
    embedding = OpenAIEmbeddings()
    global retriever
    vectordb = Chroma(persist_directory='db1', embedding_function=embedding)
    retriever = vectordb.as_retriever(search_type="mmr")


# 参考https://python.langchain.com/en/latest/modules/chains/index_examples/question_answering.html?highlight=context
def prompt(query):
    prompt_template = """You are an artificial intelligence assistant for patent analysis. The document you have received is a patent description document, and you should only use the information obtained in the context. If you do not know the answer, say "Hmm, I am not sure." Do not attempt to fabricate the answer. If the question is not about this document, please inform them politely, and you will only answer questions about this document.
    Context: {context}
    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    docs = retriever.get_relevant_documents(query)
    # 基于docs来prompt，返回你想要的内容，so easy吧！
    chain = load_qa_chain(OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=-1), chain_type="stuff",
                          prompt=PROMPT)
    result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    return result['output_text']


if __name__ == "__main__":
    # load embedding
    load_embedding()
    # 循环输入查询，直到输入 "exit"
    while True:
        query = input("Enter query (or 'exit' to quit): ")
        if query == 'exit':
            print('exit')
            break
        print("Query:" + query + '\nAnswer:' + prompt(query) + '\n')
