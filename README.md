# Fine tuning a GPT-2 model on Game Of Thrones Data

![Flow Chart](https://github.com/KaushalNaresh/AI_in_Storytelling/blob/main/Flow_Chart.png)

1. First preprocess the data
Use the notebook book_cleaning.ipynb


2. Start Training the GPT-2 model

``` 
python gpt2_finetuning.py \
    --output_dir=output \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-medium \
    --train_data_file='train_got.txt' \
    --do_eval \
    --eval_data_file='val_got.txt'\
    --overwrite_output_dir\
    --block_size=200\
    --per_gpu_train_batch_size=1\
    --model_name_or_path=gpt2-medium\
    --save_steps 5000\
    --num_train_epochs=10
  ```


## Running the APP
We have already pre-processed and fine-tuned our model so you can directly use below command (in main folder) to run the app but before that download the files given on this [link](https://www.mediafire.com/file/r4ilwku0h2ngiu1/Archive.zip/file) and add it to the main/output folder
```
export OPENAI_API_KEY="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"
streamlit run app.py
```

## Demo
Below demo is shown on Streamlit APP created.

![Demo](https://github.com/KaushalNaresh/AI_in_Storytelling/blob/main/Demo.jpg)

![Dalle_images](https://github.com/KaushalNaresh/AI_in_Storytelling/blob/main/Dalle_images.jpeg)
