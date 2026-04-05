import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
def create_embedding(text_list):
     r = requests.post("http://localhost:11434/api/embed",json={
    "model":"bge-m3",
    "input": text_list
     })
     
     embedding = r.json()["embeddings"]
     return embedding

def inference(prompt):
         r = requests.post("http://localhost:11434/api/generate",json={
    "model":"llama3.2" ,
    #"model":"deepseek-r1",
    "prompt": prompt,
     "stream":False
     })
         response = r.json()
         print(response)
         return response






df = joblib.load('embeddings.joblib')
incoming_query = input("Ask a question : ")
question_embedding = create_embedding([incoming_query])[0]


similarities = cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()
#print(similarities)
top_result = 5
max_indx = similarities.argsort()[::-1][0:top_result]
#print(max_indx)
new_df = df.loc[max_indx]
#print(new_df[["title",'number','text']])
prompt = f'''I am teaching Data structure. Here are vedios subtitle chunks containig vedio title,vedio number,start time in second,
and time in seconds, the text at that time :
{new_df[["title","number","start","end","text"]].to_json(orient= "records")}
-----------------------------
"{incoming_query}"
User asked this question related to this vedios chunks,you have to answer where and how much
content is taught in which videos where (in which vedio and at what timestamp) and guide the user to go to that 
particular video. If user asks unrelated question,tell him that you can only answer question related to the course
'''

with open("prompt.txt","w")as f:
    f.write(prompt)


response = inference(prompt)["response"]
print(response)

with open("response.txt","w")as f:
    f.write(response)


#for index,item in new_df.iterrows():
    #print(index,item["title"],item["number"],item["text"],item["start"],item["end"])