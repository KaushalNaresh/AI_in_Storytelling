# Fine tuning a GPT-2 model on your own input text

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


## Inference
We have already pre-processed and fine-tuned our model so you can 
```
export OPENAI_API_KEY="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"
streamlit run app.py
```
