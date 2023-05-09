import streamlit as st
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number

# LangChain Models
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# Standard Helpers
import pandas as pd
import requests
import time
import json
from datetime import datetime

# For token counting
import pdfplumber


def printOutput(output):
    print(json.dumps(output,sort_keys=True, indent=3))

#@st.cache(suppress_st_warning=True)
def get_toc(file):
    openai_api_key = ''
    llm = ChatOpenAI(
#     model_name="gpt-3.5-turbo", # Cheaper but less reliable
    model_name="gpt-3.5-turbo",
    temperature=0,
    max_tokens=2000,
    openai_api_key=openai_api_key)

    full_toc_schema = Object(
    id="toc",
    description="Information about the table of contents.",
    
    # Notice I put multiple fields to pull out different attributes
    attributes=[
        Text(
            id="chapter_name",
            description="The name of the chapter."
        ),
        Number(
            id="chapter_number",
            description="The number of the chapter."
        ),
        Number(
            id="start_page",
            description="The page number the chapter starts on."
        )
    ],
    examples=[
        (
            '''contents
            Preface ix
            1 The Five Domains of Digital Transformation: 
            Customers, Competition, Data, Innovation, Value 1
            2 Harness Customer Networks 19
            3 Build Platforms, Not Just Products 50
            4 Turn Data Into Assets 89
            5 Innovate by Rapid Experimentation 123
            6 Adapt Your Value Proposition 165
            7 Mastering Disruptive Business Models 194
            Conclusion 239''',
            [
                {"chapter_name": "Harness Customer Networks", "chapter_number": "2","start_page": "19"},
                {"chapter_name": "Build Platforms, Not Just Products", "chapter_number": "3","start_page": "50"},
            ],
        )
            ]
    )
    chain = create_extraction_chain(llm, full_toc_schema)
    output = chain.predict_and_parse(text=get_toc_page(file))
    return output['data']['toc']
#return the page with table of contents
def get_toc_page(file):
    
    with pdfplumber.open(file) as f:
        return f.pages[7].extract_text()


template = """
You are a specialist instructional design expert Develop workbook for each chapter of this book having following structure:

1. Big idea
2. Application in business
2. Excercises to implement the lessons
3. Summary steps for making business transformation happen

{chapter}
"""

prompt = PromptTemplate(
    input_variables=["chapter"],
    template=template
)
system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)
def get_chapter_workbook(chapter):
    openai_api_key = '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'
    llm = ChatOpenAI(  openai_api_key=openai_api_key, temperature=0.5)
    #chain = LLMChain(llm=llm, prompt_template=prompt)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

    chat_prompt_with_values = chat_prompt.format_prompt(chapter=chapter)

    response = llm(chat_prompt_with_values.to_messages()).content

    return response

def get_chapter_summary(paragraphs):
    from langchain import OpenAI
    from langchain.chains.summarize import load_summarize_chain
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import os
    openai_api_key = ''
    llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([paragraphs])
    summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce')

    summary = summary_chain.run(docs)

    return summary

def get_chapter_summary_alt(chapter):

    import torch
    
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    # Load the tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    # Load the model
    model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)

    # Set the device to CPU
    device = torch.device('cpu')
    model.to(device)

    # Tokenize the input text
    inputs = tokenizer.encode(chapter, return_tensors='pt')
    inputs.to(device)

    # Generate the summary
    outputs = model.generate(inputs, max_length=100, min_length=30)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary

if __name__ == '__main__':
    # Set page title and formatting
    st.set_page_config(page_title='BookSum - Extract PDF summaries', page_icon=':books:', layout='centered', initial_sidebar_state='auto')

    st.title('BookSum - Extract PDF summaries')
    st.subheader('Upload your PDF file')

    # Display file uploader
    file = st.file_uploader(label='', type=['pdf'])

    if file is not None:
        # Display uploaded file name and size
        st.write(f'File name: {file.name} ({round(len(file.read())/1024, 2)} KB)')
        toc = get_toc(file)

        st.subheader('Table of Contents')
        

        # Display table of contents
        index = 0
        reader = pdfplumber.open(file) 
        total_pages = len(reader.pages)

        for chapter in toc:
            index += 1
            chapter_expander = st.expander(label=chapter['chapter_number'] + '. ' + chapter['chapter_name'])
            with chapter_expander:
                st.title(f"{chapter['chapter_number']}. {chapter['chapter_name']}")
                #st.header(f"Chapter Number: {chapter['chapter_number']}: Starting Page: {chapter['start_page']}")
           # with chapter_expander:
            #    for i, section in enumerate(chapter['children']):
             #       st.write(f"{i + 1}. Page {section['page']}: {section['title']}")

            # Display chapter summary
            
            paragraph_text = ''
            
            for page_num in range(int(chapter['start_page']), int(toc[index+1]['start_page']) 
                          if index+1 < len(toc) else total_pages):
                # Extract text from chapter pages
                page = reader.pages[page_num].extract_text()
                paragraph_text += page
            with chapter_expander:
                st.subheader(f"Chapter Summary")
                chapter_summary = get_chapter_summary_alt(paragraph_text)
                st.write(chapter_summary)
                chapter_workbook = get_chapter_workbook(chapter_summary)
                st.write(chapter_workbook)
            # Split extracted text into sentences
            

                
            
