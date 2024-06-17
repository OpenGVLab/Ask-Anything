### Videochat2 accelerated using vllm

In order to increase the rate at which videochat2 generates captions, we use vllm to accelerate the videochat2 model.

##### Get started

First, you need to configure the vllm library according to the installation method of the vllm library.

https://github.com/vllm-project/vllm

Please build vllm from source from our repo, because we have some change to vllm.

Then you need to download the weights of videochat2.

https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2

To get the final language model, run

```python
python videochat2/merge_weight.py
```

Then, run vllm_test.py to accelerate video caption.

```
python videochat2/vllm_test.py
```

