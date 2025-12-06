# 471Project
An agentic framework that gives you recommendations based on your movie preferences.
Make sure you input the API Keys and Hugging face token into .env. The following are required:
HF_TOKEN=[INSERT YOUR TOKEN HERE]

OMDB_KEY=b1d90d8b

SERP_KEY=09e047a2b9fd9474fdf3dda9c97087a9d9348e0eff3a82e99d51a7a07cc25fa9
 
This project makes use of the Llama3.1-8b-Instruct model from hugging-face. On hugging face,
this model is gated, meaning you may need permission from Meta to use it. I would recommend using this model,
if possible and not changing it because other models do not perform as well. 

The code is run by the following command:

python recommendme.py
