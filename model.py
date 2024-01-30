from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA,ConversationalRetrievalChain, LLMChain
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
import chainlit as cl
from langchain.agents import AgentExecutor, Tool,initialize_agent
from langchain.agents.types import AgentType
from langchain.memory import ConversationBufferMemory, ConversationTokenBufferMemory, ConversationBufferWindowMemory


DB_FAISS_PATH = "vectorstore/db_faiss"

# custom_prompt_template ="""
# Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
# \n\nChat History:\n{chat_history}\n
# Follow Up Input: {question}\n
# Standalone question:
# """


# Please provide information from the document based on the following pieces of information. 
# If the document does not contain relevant information, respond with an indication that the answer is not available, 
# and refrain from generating responses not present in the document.


custom_prompt_template ="""
Use the following pieces of information to determine if the answer is relevant to the document.
If you don't find relevant information, please just say that you don't know the answer; don't try to make up an answer.
Avoid generating statements about the answer's relevance to the document.

Context: {context}
History: {history}
Question: {question}.

Only Output the answer below
answer:
"""
# """
# "Use the provided information to generate a response to the given question.
# If the information is insufficient or not applicable, indicate that you are unable to provide a relevant answer.
# Avoid generating statements about the answer's relevance to the document.

# Context: {context}
# History: {history}
# Question: {question}.

# Only output the answer below
# answer:"""



# custom_prompt_template = """[INST] <<SYS>>
# Please provide information from the document based on the following pieces of information. 
# If the document does not contain relevant information, respond with an indication that the answer is not available, 
# and refrain from generating responses not present in the document.

# {history}

# {context}
# <</SYS>>
# {question}
# Answer:
# [/INST]"""



# prompt_template = """
# Use the following pieces of information to determine if the answer is relevant to the document.
# If you don't find relevant information, please just say that you don't know the answer; don't try to make up an answer.

# Context: {context}
# Question: {question}

# Only return the helpful answer below if it is relevant to the document.
# Helpful answer:
# """
# condense_prompt = PromptTemplate.from_template(
#     ('Do X with user input ({question}), and do Y with chat history ({chat_history}).')
# )

# combine_docs_custom_prompt = PromptTemplate.from_template(
#     ('Write a haiku about a dolphin.\n\n'
#      'Completely ignore any context, such as {context}, or the question ({question}).')
# )
# """
# Verify the relevance of the information in the document before providing an answer.
# If the question is unrelated or the information is not present in the document, respond with "I don't know the answer."

# Context: {context}
# Question: {question}

# Provide the answer only if it is directly related to the document.
# Answer:
# """
# Conversation History:
# {conversation_history}
# Only return the helpful answer below if it is relevant to the document.
# Helpful answer:


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector stores
    """
                                                                            
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["history","context", "question"])
    
    return prompt


def load_llm():
    llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML",
                        model_type='llama', 
                        max_new_token=512,
                        temperature=0.0
                        )
    
    return llm



def retrieval_qa_chain(llm,prompt, db):
# def retrieval_qa_chain(llm, db):
    # memory = ConversationBufferMemory(memory_key="history", input_key="question")
    # memory = ConversationTokenBufferMemory(memory_key="history", input_key="question", max_token_limit=200,llm=llm)
    memory = ConversationBufferWindowMemory(memory_key="history", input_key="question", k=2,llm=llm)
    retriever = db.as_retriever(search_kwargs={"k": 2,"score_threshold": 0.1},search_type="similarity_score_threshold")

    qa_chain = RetrievalQA.from_chain_type(
        llm= llm,
        chain_type= "stuff",
        verbose=True,
        retriever= retriever,
        return_source_documents= True,
        chain_type_kwargs= {"prompt": prompt, "memory" : memory}
    )
    
    # qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, 
    #                                                 retriever=retriever,
    #                                                 memory=memory,
    #                                                 chain_type="stuff",
    #                                                 verbose=True,
    #                                                 # combine_docs_chain_kwargs={"prompt": prompt},
    #                                                 condense_question_prompt=condense_prompt,
    #                                                 combine_docs_chain_kwargs=dict(prompt=combine_docs_custom_prompt)
    #                                                 )
    # qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff", retriever=retriever, memory=memory)

    return qa_chain

    


       
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', 
                                       model_kwargs={'device' : 'cpu'})
    
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm=llm, prompt=qa_prompt, db=db)


    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query' : query})
    return response




# if __name__ == "__main__":
#     st.header("Knowledge based Chatbot")

#     input_text = st.text_input("Enter Your Question")

#     submit = st.button("Generate")

#     ### Final Reponse
#     if submit:
#         output = final_result(input_text)
#         st.write(output["result"])
# chat_history = []
@cl.on_chat_start
async def start():
    chain=qa_bot()
    # print("Chain is ",  chain)
    msg=cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content="Hi, Welcome to the Chat Bot. What is your query?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        # stream_final_answer= True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached=True
    res = await chain.acall(message.content, callbacks=[cb])
    print("Response is: ", res)
    answer= res["result"]       
    sources = res["source_documents"]

    print("Response is: ", res)
    if sources:
        answer = f"\n{str(answer)}"
    else:
        answer = "No context found"

    await cl.Message(content=answer).send()



