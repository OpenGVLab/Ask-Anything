from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
import os
import re
import gradio as gr
import openai

# Supported LLM providers and their default models
LLM_PROVIDERS = {
    "openai": {
        "default_model": "gpt-4",
        "api_base": None,  # uses default OpenAI endpoint
    },
    "minimax": {
        "default_model": "MiniMax-M3",
        "api_base": "https://api.minimax.io/v1",
    },
}


def create_llm(provider, api_key, model_name=None, temperature=0):
    """Create an LLM instance based on the selected provider.

    Args:
        provider: LLM provider name ("openai" or "minimax").
        api_key: API key for the chosen provider.
        model_name: Model name override.  Uses provider default when None.
        temperature: Sampling temperature.

    Returns:
        A LangChain LLM or ChatModel instance.
    """
    provider = provider.lower()
    if provider not in LLM_PROVIDERS:
        raise ValueError(
            f"Unsupported provider '{provider}'. "
            f"Supported: {list(LLM_PROVIDERS.keys())}"
        )

    cfg = LLM_PROVIDERS[provider]
    model = model_name or cfg["default_model"]

    if provider == "minimax":
        # MiniMax requires temperature in (0.0, 1.0]
        temperature = max(0.01, min(temperature, 1.0))
        return ChatOpenAI(
            model_name=model,
            openai_api_key=api_key,
            openai_api_base=cfg["api_base"],
            temperature=temperature,
        )

    # Default: OpenAI
    return OpenAI(
        temperature=temperature,
        openai_api_key=api_key,
        model_name=model,
    )


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


    def init_agent(self, api_key, image_caption, dense_caption, video_caption, tags, state, provider="openai"):
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

        # Resolve provider from argument or environment
        provider = (provider or os.environ.get("LLM_PROVIDER", "openai")).lower()

        if not api_key or not api_key.strip():
            return gr.update(visible=False), state, state, "Please paste your API key!"

        # Provider-specific API key validation
        if provider == "openai" and not api_key.startswith("sk-"):
            return gr.update(visible=False), state, state, "Please paste your OpenAI key (sk-...)!"

        self.llm = create_llm(provider=provider, api_key=api_key)
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS, 'suffix': SUFFIX}, )
        state = state + [("I upload a video, Please watch it first! ","I have watch this video, Let's chat!")]
        return gr.update(visible = True),state, state, api_key

if __name__=="__main__":
    import pdb
    pdb.set_trace()
