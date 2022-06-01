FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
ADD . ru_gpt2
RUN python -m pip install -r ru_gpt2/requirements.txt
RUN python -m spacy download ru_core_news_sm