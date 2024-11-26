from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
import re
import gradio as gr
import openai


def cut_dialogue_history(history_memory, keep_last_n_words=400):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split('\n')
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(' '))
        paragraphs = paragraphs[1:]
    return '\n' + '\n'.join(paragraphs)


class ConversationBot:
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')
        self.tools = []

    def run_text(self, text, state):
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text.strip()})
        res['output'] = res['output'].replace("\\", "/")
        response = res['output'] 
        state = state + [(text, response)]
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.buffer}")
        return state, state


    def init_agent(self, openai_api_key, image_caption, dense_caption, video_caption, tags, state):
        chat_history =''
        PREFIX = "ChatVideo is a chatbot that chats with you based on video descriptions."
        FORMAT_INSTRUCTIONS = """
        When you have a response to say to the Human,  you MUST use the format:
        ```
        {ai_prefix}: [your response here]
        ```
        """
        SUFFIX = f"""You are a chatbot that conducts conversations based on video descriptions. You mainly answer based on the given description, and you can also modify the content according to the tag information, and you can also answer the relevant knowledge of the person or object contained in the video. The second description is a description for one second, so that you can convert it into time. When describing, please mainly refer to the sceond description. Dense caption is to give content every five seconds, you can disambiguate them in timing. But you don't create a video plot out of nothing.

                Begin!

                Video tags are: {tags}

                The second description of the video is: {image_caption}

                The dense caption of the video is: {dense_caption}

                The general description of the video is: {video_caption}"""+"""Previous conversation history {chat_history}

                New input: {input}

                {agent_scratchpad}
                """
        self.memory.clear()
        if not openai_api_key.startswith('sk-'):
            return gr.update(visible = False),state, state, "Please paste your key here !"
        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key,model_name="gpt-4")
        # openai.api_base = 'https://api.openai-proxy.com/v1/'  
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS, 'suffix': SUFFIX}, )
        state = state + [("I upload a video, Please watch it first! ","I have watch this video, Let's chat!")]
        return gr.update(visible = True),state, state, openai_api_key

if __name__=="__main__":
    import pdb
    pdb.set_trace()
