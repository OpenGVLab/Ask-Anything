import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList
import time
import numpy as np
from torch.nn import functional as F
import os

# auth_key = os.environ["HF_ACCESS_TOKEN"]

start_message = """<|SYSTEM|># StableAssistant
- StableAssistant is A helpful and harmless Open Source AI Language Model developed by Stability and CarperAI.
- StableAssistant is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableAssistant is more than just an information source, StableAssistant is also able to write poetry, short stories, and make jokes.
- StableAssistant will refuse to participate in anything that could harm a human."""


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class StableLMBot:
    def __init__(self):
        print(f"Starting to load the model to memory")
        self.m = AutoModelForCausalLM.from_pretrained(
            "stabilityai/stablelm-tuned-alpha-7b", torch_dtype=torch.float16).cuda()
        self.tok = AutoTokenizer.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")
        self.generator = pipeline('text-generation', model=self.m, tokenizer=self.tok, device=0)
        self.messages = start_message
        print(f"Sucessfully loaded the model to the memory")

    def contrastive_generate(self,text, bad_text):
        with torch.no_grad():
            tokens = self.tok(text, return_tensors="pt")[
                'input_ids'].cuda()[:, :4096-1024]
            bad_tokens = self.tok(bad_text, return_tensors="pt")[
                'input_ids'].cuda()[:, :4096-1024]
            history = None
            bad_history = None
            curr_output = list()
            for i in range(1024):
                out = self.m(tokens, past_key_values=history, use_cache=True)
                logits = out.logits
                history = out.past_key_values
                bad_out = self.m(bad_tokens, past_key_values=bad_history,
                            use_cache=True)
                bad_logits = bad_out.logits
                bad_history = bad_out.past_key_values
                probs = F.softmax(logits.float(), dim=-1)[0][-1].cpu()
                bad_probs = F.softmax(bad_logits.float(), dim=-1)[0][-1].cpu()
                logits = torch.log(probs)
                bad_logits = torch.log(bad_probs)
                logits[probs > 0.1] = logits[probs > 0.1] - bad_logits[probs > 0.1]
                probs = F.softmax(logits)
                out = int(torch.multinomial(probs, 1))
                if out in [50278, 50279, 50277, 1, 0]:
                    break
                else:
                    curr_output.append(out)
                out = np.array([out])
                tokens = torch.from_numpy(np.array([out])).to(
                    tokens.device)
                bad_tokens = torch.from_numpy(np.array([out])).to(
                    tokens.device)
            return self.tok.decode(curr_output)


    def generate(self, text, bad_text=None):
        stop = StopOnTokens()
        result = self.generator(text, max_new_tokens=1024, num_return_sequences=1, num_beams=1, do_sample=True,
                        temperature=1.0, top_p=0.95, top_k=1000, stopping_criteria=StoppingCriteriaList([stop]))
        return result[0]["generated_text"].replace(text, "")
    
    def init_agent(self, image_caption, dense_caption, tags):
        SUFFIX = f"""You are a chatbot that conducts conversations based on video descriptions. You mainly answer based on the given description, and you can also modify the content according to the tag information, and you can also answer the relevant knowledge of the person or object contained in the video. The second description is a description for one second, so that you can convert it into time. When describing, please mainly refer to the sceond description. Dense caption is to give content every five seconds, you can disambiguate them in timing. But you don't create a video plot out of nothing.

                Begin!

                Video tags are: {tags}

                The second description of the video is: {image_caption}

                The dense caption of the video is: {dense_caption}"""
        
        self.messages = start_message + SUFFIX
        return gr.update(visible = True), [("I have uploaded a video, please watch it!","Ask me!")]

    def run_text(self, text, state):
        state.append([text,""])
        history = self.messages
        for content in state:
            history = history + "<|USER|>" + content[0] + "<|ASSISTANT|>" + content[1]
        output = self.generate(history)
        state[-1][-1] = output
        history = history + output
        return state,state

# bot = StableLMBot()