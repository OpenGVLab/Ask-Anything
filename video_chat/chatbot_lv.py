from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory,ConversationBufferWindowMemory
from langchain.llms.openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import ChatVectorDBChain
from langchain.prompts.prompt import PromptTemplate
import re
import gradio as gr
import openai



class ConversationBot:
    def __init__(self):
        self.memory = [] 
        self.tools = []

    def run_text(self, text, state):
        res = self.agent({"question": text.strip(),"chat_history": self.memory})
        ans = res['answer'].strip("\nAnswer:")
        # print(res)
        # print(res['answer'])
        self.memory.append((text,ans))
        state = state + [(text, ans)]
        return state, state


    def init_agent(self, openai_api_key, image_caption, dense_caption, video_caption, tags, state,subtitile):
        SUFFIX = """You are a chatbot that conducts conversations based on video contexts. You mainly answer based on the given contexts, and you can also modify the content according to the tag information, and you can also answer the relevant knowledge of the person or object contained in the video. The timing description is a description every one second, so that you can convert it into time. When describing, please mainly refer to the timing description. Dense caption is to give content every five seconds, you can disambiguate them in timing. But you don't create a video plot out of nothing.
                Begin!

                Video contexts in temporal order:{context}

                Question:{question}
                """
        self.memory = []
        print(SUFFIX)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
        if not openai_api_key.startswith('sk-'):
            return gr.update(visible = False),state, state, "Please paste your key here !"
        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")
        documents =f"""Video tags are: {tags}

                The second description of the video is: {image_caption}

                The general description of the video is: {video_caption}
                
                The subtitile of the video is: {subtitile}"""+"""Previous conversation history {chat_history}
                                
                """


        # documents =f"""

        #         The conversations in the video are:\n {subtitile}

        #         """+"""\nPrevious conversation history {chat_history}
                                
        #         """
                 
                 
        self.faissvector = FAISS.from_documents(self.text_splitter.create_documents([documents]), OpenAIEmbeddings())
        
        condense_question_prompt = PromptTemplate(input_variables=['chat_history', 'question'], output_parser=None, partial_variables={}, template='Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone question:', template_format='f-string', validate_template=True)
        qa_prompt = PromptTemplate(template=SUFFIX, input_variables=["context","question"])
        self.agent = ChatVectorDBChain.from_llm(self.llm,
                                                self.faissvector,
                                                condense_question_prompt=condense_question_prompt,
                                                qa_prompt=qa_prompt)

        state = state + [("I upload a video, Please watch it first! ","I have watch this video, Let's chat!")]
        return gr.update(visible = True), state, state, openai_api_key
