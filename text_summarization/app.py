import streamlit as st
import validators
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain.chains.summarize import load_summarize_chain

st.set_page_config(page_title="LangChain: Summarize Text from website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From Website")
st.subheader('Summarize URL')

## Get the Groq API Key and url to be summarized
with st.sidebar:
    groq_api_key=st.text_input("Provide Your Groq key",value="",type="password")

if groq_api_key:
    generic_url=st.text_input("URL",label_visibility="collapsed")

    ## Gemma Model USsing Groq API
    llm =ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

    prompt_template="""
    Provide a summary of the following content in 300 words:
    Content:{text}
    """
    prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

    if st.button("Summarize the Content Website"):
        if not validators.url(generic_url):
         st.error("Please enter a valid website URL")
        else:
         try:
            with st.spinner("Waiting..."):
              if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
              else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                    headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    docs=loader.load()

                    ## Chain For Summarization
                    chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                    output_summary=chain.run(docs)

                    st.success(output_summary)
              
         except Exception as e:
            st.exception(f"Exception:{e}")
              

    
else:
    st.error("Please provide your groq key")