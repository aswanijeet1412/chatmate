import streamlit as st
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

import os
import urllib.parse
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain

#function to fetch text data from the links of news websites
# def fetch_article_content(url):
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
#     }
#     try:
#         response = requests.get(url, headers=headers)
#         response.raise_for_status()
#         return response.text
#     except requests.RequestException as e:
#         st.error(f"Error fetching {url}: {e}")
#         return ""

#function to collate all the text from the news website into a single string
# def process_links(links):
#     all_contents = ""
#     for link in enumerate(links):
#        # link=urllib.parse.urljoin(link, '/')
#         loader = WebBaseLoader(link)
#         #documents = loader.load()
#         content = loader.load()
#         content=str(content)
#         content=content.strip()
#         all_contents += content + "\n\n"
#     return all_contents

#function to chunk the articles beofore creating vector embeddings
# def get_text_chunks_langchain(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     texts = text_splitter.split_text(text)
#     return texts

#creating the streamlit app

def main():
    #st.title('Nebo :  News article Bot')
    
   filename = "chatmate.png"
   # img = cv2.imread(filename, 1)
   # image = np.array([img])
   

   left_co, cent_co,last_co = st.columns(3)
   with cent_co:
       new_title = '<p style="font-family:Courier; color:Blue; font-size: 37px;">Chatmate</p>'
       st.markdown(new_title, unsafe_allow_html=True)
       st.image(filename,width=150, channels="BGR")
       
       
   sub_title= '<p style="font-family:Courier; color:Black; font-size: 20px;"> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;Chat with Chatmate! </p>'
   st.markdown(sub_title, unsafe_allow_html=True)
   
   if 'articles_fetched' not in st.session_state:
       st.session_state.articles_fetched = False
   if 'chat_history' not in st.session_state:
       st.session_state.chat_history = ""
      # upload a PDF file
     
    #st.header(displayimage(image_placeholder, imagePath,"Chatty"))
   # imagePath = "https://storage.prompt-hunt.workers.dev/clfmmjo190005l60880y1f4g0_2"
    #imagePath2 = "/PathToUsersFolder/Rotating_earth_(large).gif"
    #displayimage(image_placeholder, imagePath,"")

    # Sidebar contents
   with st.sidebar:
        
        

        # # Initialize state variables
        # if 'articles_fetched' not in st.session_state:
        #     st.session_state.articles_fetched = False
        # if 'chat_history' not in st.session_state:
        #     st.session_state.chat_history = ""
            
        # # Model selection
        
        # with st.sidebar:
        model_choice = st.radio("Choose your model", ["GPT 3.5", "GPT 4"], key= "model_choice")
        model = "gpt-3.5-turbo-1106" if model_choice == "GPT 3.5" else "gpt-4-1106-preview"
        API_KEY = st.text_input("Enter your OpenAI API key", type="password", key= "API_KEY")
        #os.environ["OPENAI_API_KEY"] =  st.session_state.API_KEY
            

            
            # Ensure API_KEY is set before proceeding
            
        st.warning("Please enter your OpenAI API key.")
        pdf = st.file_uploader("Upload your PDF", type='pdf')
            
        st.markdown('''
            ## About
            This is OpenAI based chatbot built by: 
            [Jitendra Aswani](https://www.linkedin.com/in/jitendra-aswani-72800216/)''')

 

    #API_KEY
    

    

    #asking user to upload a text file with links to news articles (1 link per line)
    

    # Read the file into a list of links
# =============================================================================
#     if uploaded_file:
#         stringio = uploaded_file.getvalue().decode("utf-8")
#         link = stringio.splitlines()
# =============================================================================

    # Fetch the articles' content
# =============================================================================
#     if st.button("Fetch Articles") and uploaded_file:
#         with st.spinner('Fetching articles...'):
# =============================================================================
            # article_contents = process_links(links)
        if pdf is not None:
            pdf_reader = PdfReader(pdf)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                texts = text_splitter.split_text(text)
                os.environ["OPENAI_API_KEY"] =  st.session_state.API_KEY
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(texts, embedding=embeddings)
                #vector_store = Qdrant.from_texts(texts, embeddings, location=":memory:",)
                #retriever = vector_store.as_retriever()
                    #Creating a QA chain against the vectorstore
                llm = ChatOpenAI(model_name= model)
                #chain = load_qa_chain(llm=llm, chain_type="stuff")
                if 'qa' not in st.session_state:
                        st.session_state.qa = load_qa_chain(llm=llm, chain_type="stuff")
                    

                #st.success('Articles fetched successfully!')
                st.session_state.articles_fetched = True
                 

            #Process the article contents
        
            

            #storing the chunked articles as embeddings in Qdrant
       

    #once articles are fetched, take input for user query

   if 'articles_fetched' in st.session_state and st.session_state.articles_fetched:

        query = st.text_input("Enter your query here:", key="query")

        if query:
            # Process the query using your QA model (assuming it's already set up)
            with st.spinner('Analyzing query...'):
                qa = st.session_state.qa
                docs = VectorStore.similarity_search(query=query, k=3)
                response = st.session_state.qa.run(input_documents=docs, question=query)
                #response = qa.run(st.session_state.query)  
            # Update chat history
            st.session_state.chat_history += f"> {st.session_state.query}\n{response}\n\n"

        # Display conversation history
        st.text_area("Conversation:", st.session_state.chat_history, height=1000, key="conversation_area")
        # JavaScript to scroll to the bottom of the text area
        st.markdown(
            f"<script>document.getElementById('conversation_area').scrollTop = document.getElementById('conversation_area').scrollHeight;</script>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
