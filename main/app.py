import os
import requests
from io import BytesIO
import openai
import PIL.Image
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import logging
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import trange
import streamlit as st
from gtts import gTTS
from run import kpfilter
import requests
from io import BytesIO
import PIL.Image
import PIL.ImageEnhance
from typing import List
import base64
# Main function
trn_mdl = {'gpt2': (GPT2LMHeadModel, GPT2Tokenizer)}
logger = logging.getLogger(__name__)
max_val = int(998)
openai.api_key = os.environ["OPENAI_API_KEY"]

def text_to_speech(text, audio_filename):
    tts = gTTS(text, lang='en')
    tts.save(audio_filename)

def generate_images(prompt: str, api_key: str, num_images: int = 3) -> List[str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "image-alpha-001",
        "prompt": prompt,
        "num_images": num_images,
        "size": "256x256",
        "response_format": "url",
    }
    response = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=data)
    response.raise_for_status()
    return [result["url"] for result in response.json()["data"]]

def enhance_image(image: PIL.Image.Image) -> PIL.Image.Image:
    # Apply post-processing techniques to enhance the generated image
    enhancer = PIL.ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.2)
    enhancer = PIL.ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.2)
    return image


def seqsample(outp, tot_seq, point, tot_count=1, tmp=1, tk=0, tp=0.0, no_rep=1.0, dev_type='cpu'):
    point = torch.tensor(point, dtype=torch.long, device=dev_type)
    point = point.unsqueeze(0).repeat(tot_count, 1)
    res_received = point
    with torch.no_grad():
        for z in trange(tot_seq):
            inval = {'input_ids': res_received}
            res = outp(**inval)
            prob_val_prov = res[0][:, -1, :] / (tmp if tmp > 0 else 1.)
            for i in range(tot_count):
                for _ in set(res_received[i].tolist()):
                    prob_val_prov[i, _] /= no_rep
            prob_val_arranged = kpfilter(prob_val_prov, tk=tk, tp=tp)
            if tmp == 0:
                gen_val = torch.argmax(prob_val_arranged, dim=-1).unsqueeze(-1)
            else:
                gen_val = torch.multinomial(F.softmax(prob_val_arranged, dim=-1), num_samples=1)
            res_received = torch.cat((res_received, gen_val), dim=1)
    return res_received

def generate_text_with_custom_code(prompt, args):
    args.dev_type = torch.device("cuda" if torch.cuda.is_available() and not args.gpu_cpu else "cpu")
    args.gpu_val = torch.cuda.device_count()
    seed(args)
    args.model = args.model.lower()
    model_class, tokenizer_class = trn_mdl[args.model]
    tokenizer = tokenizer_class.from_pretrained(args.path_to_model)
    outp = model_class.from_pretrained(args.path_to_model)
    outp.to(args.dev_type)
    outp.eval()

    if args.tot_seq < 0 and outp.config.max_position_embeddings > 0:
        args.tot_seq = outp.config.max_position_embeddings
    elif 0 < outp.config.max_position_embeddings < args.tot_seq:
        args.tot_seq = outp.config.max_position_embeddings 
    elif args.tot_seq < 0:
        args.tot_seq = max_val

    tknsvals = tokenizer.encode(prompt, add_special_tokens=False)

    results = seqsample(
        outp=outp,
        point=tknsvals,
        tot_count=args.Total_samples,
        tot_seq=args.tot_seq,
        tmp=args.tmp,
        tk=args.tk,
        tp=args.tp,
        no_rep=args.no_rep,
        dev_type=args.dev_type)
    results = results[:, len(tknsvals):].tolist()

    obtained_texts = []
    for r in results:
        obtained = tokenizer.decode(r, clean_up_tokenization_spaces=True)
        if args.cut_gen:
            obtained = obtained[: obtained.find(args.cut_gen)]
        obtained_texts.append(obtained)

    return obtained_texts[0]


def seed(args):
    torch.manual_seed(args.dataval)
    np.random.seed(args.dataval)
    if args.gpu_val > 0:
        torch.cuda.manual_seed_all(args.dataval)
# Remaining code...

# Streamlit app
st.title("GPT-2 Text Generation for Game of Thrones")
page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''

st.markdown(
    """
<style>
    body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
    }

    .slideshow {
        width: 100%;
        height: 400px;
        overflow: hidden;
        position: relative;
        box-sizing: border-box;
    }

    .slideshow img {
        width: 100%;
        height: 100%;
        position: absolute;
        opacity: 0;
        transition: opacity 1s ease-in-out;
        -o-transition: opacity 1s ease-in-out;
        -moz-transition: opacity 1s ease-in-out;
        -webkit-transition: opacity 1s ease-in-out;
    }

    .slideshow img.show {
        opacity: 1;
    }

    .title, .markdown-text-container p, label, .st-cb, .st-d8, .st-cc {
        color: white !important;
    }

    .st-ij {
        background-color: rgba(30, 30, 30, 0.8) !important;
    }
</style>

<script>
    function slideSwitch() {
        var slideShow = document.getElementsByClassName("slideshow")[0];
        var current = slideShow.getElementsByClassName("show")[0];
        current.classList.remove("show");
        if (current.nextElementSibling) {
            current.nextElementSibling.classList.add("show");
        } else {
            slideShow.getElementsByTagName("img")[0].classList.add("show");
        }
    }
    setInterval(slideSwitch, 3000);
</script>
""",
    unsafe_allow_html=True,
)


user_input = st.text_input("Your Prompt")

# Add the Generate button
generate_button = st.button("Generate")

if user_input and generate_button:
    args = argparse.Namespace(
        path_to_model="output",
        model="gpt2",
        text="",
        tot_seq=200,
        Total_samples=1,
        textpad="",
        tmp=0.8,
        no_rep=1.0,
        tp=0.9,
        gpu_cpu=False,
        tk=0,
        dataval=42,
        cut_gen=None,
    )
    with st.spinner("Generating text..."):
        generated_text = generate_text_with_custom_code(user_input, args)
    st.write(generated_text)

    # Modify the preprocessed_text before sending it to the DALL-E function
    dalle_prompt = f"{generated_text} in the style of Game of Thrones"

    # Generate and display multiple images using DALL-E
    with st.spinner("Generating images..."):
        image_urls = generate_images(dalle_prompt, api_key=openai.api_key, num_images=4)
        images = []
        for image_url in image_urls:
            image_response = requests.get(image_url)
            image = PIL.Image.open(BytesIO(image_response.content))

            # Enhance the image
            enhanced_image = enhance_image(image)
            images.append(enhanced_image)
        
        st.image(images, caption=["DALL-E generated image"] * len(images), use_column_width=True)
        audio_file = "temp_audio.wav"
        text_to_speech(generated_text, audio_file)
        st.audio(audio_file)
        os.remove(audio_file)