from PIL import Image

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from enum import auto, Enum

import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + ": " + message + conv.sep
        else:
            ret += role + ":"
    return ret


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


class Chat:
    def __init__(self, model, device='cuda:0'):
        self.device = device
        self.model = model
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    def ask(self,text,conv):
        conv.messages.append([conv.roles[0], text + '\n'])
        return conv

    def answer(self, conv, img_list, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0):
        conv.messages.append([conv.roles[1], None])
        embs = self.get_context_emb(conv, img_list)
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=self.stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
                output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
                output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
        conv.messages[-1][1] = output_text
        return output_text, output_token.cpu().numpy(), conv
        
    def get_index(self, num_frames, num_segments):
        seg_size = float(num_frames - 1) / num_segments
        start = int(seg_size / 2)
        offsets = np.array([
            start + int(np.round(seg_size * idx)) for idx in range(num_segments)
        ])
        return offsets

    def load_video(self, video_path, num_segments=8, return_msg=False):
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)
        frame_indices = self.get_index(num_frames, num_segments)
        
        #duration = len(vr) // vr.get_avg_fps()
        #index = np.linspace(0, len(vr)-1, num=int(duration))
        # transform
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        
        transform = T.Compose([
            GroupScale(int(224), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(224),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])

        images_group = list()
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        torch_imgs_224 = transform(images_group)
        if return_msg:
            fps = float(vr.get_avg_fps())
            sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
            # " " should be added in the start and end
            msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
            return torch_imgs_224, msg
        else:
            return torch_imgs_224
        
    def get_sinusoid_encoding_table(self, n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784): 
        ''' Sinusoid position encoding table ''' 
        # TODO: make it with torch instead of numpy 
        def get_position_angle_vec(position): 
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
        
        # generate checkpoint position embedding
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
        sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
        
        print(f"n_position: {n_position}")
        print(f"pre_n_position: {pre_n_position}")
        
        if n_position != pre_n_position:
            T = ckpt_num_frame # checkpoint frame
            P = 14 # checkpoint size
            C = d_hid
            new_P = int((n_position // cur_frame) ** 0.5) # testing size
            if new_P != 14:
                print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
                print(f'Interpolate the position embedding')
                sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
                sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
                sinusoid_table = torch.nn.functional.interpolate(
                    sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
                sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
        
        if cur_frame != ckpt_num_frame:
            print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
            print(f'Interpolate the position embedding')
            T = ckpt_num_frame # checkpoint frame
            new_T = cur_frame # testing frame
            # interpolate
            P = int((n_position // cur_frame) ** 0.5) # testing size
            C = d_hid
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
            sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
            sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
        return sinusoid_table

    def upload_video(self, image, conv, img_list, num_segments):
        if isinstance(image, str):  # is a image path
            vid, msg = self.load_video(image, num_segments=num_segments, return_msg=True)
            TC, H, W = vid.shape
            video = vid.reshape(1, TC//3, 3, H, W).to(self.device)
        else:
            raise NotImplementedError
        print("Input video shape:", vid.shape)
        new_pos_emb = self.get_sinusoid_encoding_table(n_position=(224//16)**2*num_segments, cur_frame=num_segments)
        self.model.vision_encoder.encoder.pos_embed = new_pos_emb
        image_emb, _ = self.model.encode_img(video, "Watch the video and answer the question.")
        img_list.append(image_emb)
        conv.messages.append([
            conv.roles[0], 
            f"<Video><VideoHere></Video>\n"
        ])
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg, img_list, conv
    
    def upload_img(self, image, conv, img_list):
        img = image#Image.open(image)#.convert('RGB')
        transform = T.Compose(
            [
                T.Resize(
                    (224, 224), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

        img = transform(img).unsqueeze(0).unsqueeze(0).cuda()
        image_emb, _ = self.model.encode_img(img, "Observe the image and answer the question.")
        img_list.append(image_emb)
        conv.messages.append([
            conv.roles[0],
            f"<Image><ImageHere></Image>\n"
        ])
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg,img_list, conv

    def get_context_emb(self, conv, img_list):
        prompt = get_prompt(conv)
        #print(prompt)
        if '<VideoHere>' in prompt:
            prompt_segs = prompt.split('<VideoHere>')
        else:
            prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of visual placeholders and videos."
        with torch.no_grad():
            seg_tokens = [
                self.model.llama_tokenizer(
                    seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
                # only add bos to the first seg
                for i, seg in enumerate(prompt_segs)
            ]
            seg_embs = [self.model.llama_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

