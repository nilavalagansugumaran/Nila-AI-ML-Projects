# RAG (Retrieval Augumented Generation) - Example
RAG is a mechanism of retrieving and incorporating external information before LLM generates a response. The external information has to be vector embeddings usually stored in a vector database such as pinecore, opensearch etc... This techinique is very cost and time effective comparing with fine tuning model in which the LLM model will need to be re-trained.

There are ML models available which helps creating embeddings from text or images or both(multimodel models).

In this example, we are going to expore a typical workflow of RAG.

- Step 1: Read document for embedding
- Step 2: Create embedding using amazon embedding model - amazon.titan-embed-text-v2:0
- Step 3: Create a dataframe to store the embeddings (mock vector database)
- Step 4: Retrieve similar document for a prompt - Using Cosign Similarity (prompt and embedding)
- Step 5: Create prompt for the LLM along with the context emmbedding
- Step 6: Generate content using the Meta Llama model - ``

## Step 1: Read documents for embedding
We will use Pypdf to read a PDF document. For re-usability purpose, lets create a function to read a pdf file.


```python
# Use pypdf to read PDF document as string
# Install pypdf if not exists already
!pip install pypdf
```

    Looking in indexes: https://pypi.python.org/simple
    Requirement already satisfied: pypdf in /Users/nila/Documents/Learning.nosync/AI/Nila-AI-ML-Projects/bedrock_samples/env/lib/python3.13/site-packages (5.5.0)



```python
from pypdf import PdfReader

#Lets create a function to read PDF pages

def as_text(pdf_file_name):
    reader = PdfReader(pdf_file_name)
    print(f'Number of pages {len(reader.pages)}')
    # Lets read only the first page
    page = reader.pages[0]
    return page.extract_text()

#print(as_text('sports_news.pdf'))
```

## Step 2: Create embedding using amazon embedding model
We are going to use Amazon embedding model - amazon.titan-embed-text-v2:0

Request format for the model is as below,
```
{
    "inputText": string,
    "dimensions": int,
    "normalize": boolean,
    "embeddingTypes": list
}
```
Response format,
```
{
    "embedding": [float, float, ...],
    "inputTextTokenCount": int,
    "embeddingsByType": {"binary": [int,..], "float": [float,...]}
}
```
Reference documentation - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html



```python
# Configure AWS environment
# Below method is NOT recommended for public sharing or deployments to environments
import os

os.environ['AWS_ACCESS_KEY_ID'] = 'your key id'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your secret'

os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
```


```python
# Lets use boto3 to connect to bedrock APIs to access the model
import boto3
# Use json to build the request
import json
```


```python
# For reuseability lets create a function
def get_text_embedding(text_for_embedding):
    bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
    req = json.dumps({'inputText':text_for_embedding})
    response = bedrock_runtime.invoke_model(body=req, modelId='amazon.titan-embed-text-v2:0')
    response_body = json.loads(response.get('body').read())
    return response_body['embedding']
#print(get_text_embedding(as_text('sports_news.pdf')))

```

## Step 3: Create a data frame to store the embeddings (mock vector database)¶
In production like environments, the embeddings will be stored in a vector database. For the purpose of demonstration, lets create a mock vector database using pandas dataframe. The dataframe need to have the below,

- Identifier/title
- Text
- Embedding
-------------------------------
identifier | Text | Embedding
-------------------------------
       |               |
       |               |
       |               |       
-------------------------------

Lets use the functions that we have created on steps 1 and 2 to create the mock vector database.



```python
# Use pandas to create dataframe
import pandas as pd

sports_news = as_text('sports_news.pdf')
employment_news = as_text('uk-job-market-news.pdf')
data = {'identifier':['cricket-news', 'uk-employment-news'],
       'text':[sports_news,employment_news],
        'embeddings':[get_text_embedding(sports_news), get_text_embedding(employment_news)]
       }
vector_db = pd.DataFrame(data)
vector_db
```

    Ignoring wrong pointing object 6 0 (offset 0)
    Ignoring wrong pointing object 6 0 (offset 0)


    Number of pages 1
    Number of pages 1





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>identifier</th>
      <th>text</th>
      <th>embeddings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cricket-news</td>
      <td>Legendary India batter Virat Kohli has announc...</td>
      <td>[-0.03001970238983631, 0.026122447103261948, 0...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>uk-employment-news</td>
      <td>The UK's job market has continued to weaken wi...</td>
      <td>[0.023377733305096626, 0.07752429693937302, -0...</td>
    </tr>
  </tbody>
</table>
</div>



## Step 4: Retrieve similar document for a prompt - Using Cosign Similarity (prompt and embedding)

In order to send the prompt and revelant context to the LLM, we look up for similar content from vector database. The method used here is **Codign Similarity** 

The cosine similarity formula measures the similarity between two vectors by calculating the cosine of the angle between them. It's defined as the dot product of the two vectors divided by the product of their magnitudes (Euclidean norms). The formula can be represented as:
```
cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)
```
**Where:**

- (A · B): is the dot product of vectors A and B.
- ||A|| and ||B|| are the Euclidean norms (magnitudes) of vectors A and B

**This formula results in a value between -1 and 1, where:**

- A value of 1 indicates that the vectors have the same direction and are perfectly similar.
- A value of 0 indicates that the vectors are perpendicular and have no similarity.
- A value of -1 indicates that the vectors have opposite directions and are perfectly dissimilar.



```python
# Lets create a function to calculate cosign similarity
import numpy as np
```


```python
def cosign_similarity(vector1, vector2):
    vec1_array = np.array(vector1)
    vec2_array = np.array(vector2)
    dot_product = np.dot(vec1_array, vec2_array)
    magnitude_vec1 = np.linalg.norm(vec1_array)
    magnitude_vec2 = np.linalg.norm(vec2_array) 
    return dot_product/(magnitude_vec1 * magnitude_vec2)

#cosign_similarity(vector_db['embeddings'][0],vector_db['embeddings'][1])
```

## Step 5: Create prompt for the LLM along with the context emmbedding
Lets find similar text (context) from the vector DB for the given prompt by comparing vectors.


```python

prompt = "Has the UK unemployment rate increased or decreased recently?"

# Create text embedding for the prompt
embedding_for_prompt = get_text_embedding(prompt)

# Find cosign similarity for each item in the vector database and get the most similar text

vector_db['prompt_similarity_score'] = vector_db['embeddings'].apply(lambda vec : cosign_similarity(vec, embedding_for_prompt))
vector_db

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>identifier</th>
      <th>text</th>
      <th>embeddings</th>
      <th>prompt_similarity_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>cricket-news</td>
      <td>Legendary India batter Virat Kohli has announc...</td>
      <td>[-0.03001970238983631, 0.026122447103261948, 0...</td>
      <td>0.038281</td>
    </tr>
    <tr>
      <th>1</th>
      <td>uk-employment-news</td>
      <td>The UK's job market has continued to weaken wi...</td>
      <td>[0.023377733305096626, 0.07752429693937302, -0...</td>
      <td>0.557581</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Get the most similar text by comparing the prompt_similarity_score 
most_similar_text = vector_db.nlargest(1, 'prompt_similarity_score').iloc[0]['text']
#most_similar_text
```


```python
full_prompt = f"Answer the question based on the context provided.\n Question: {prompt}\n And the context: {most_similar_text}\n"
#full_prompt
```

## Step 6: Generate content using LLM 

LLM Model to use - Meta Llama - `meta.llama3-70b-instruct-v1:0`

Lets use https://github.com/nilavalagansugumaran/Nila-AI-ML-Projects/blob/main/bedrock_samples/meta-llama/meta-llama-text-generation.md as a reference to construct the request and generate content using the LLM


```python
# Use json module to construct the requets payload
import json
```


```python
# Embed the prompt in Llama 3's instruction format.
prompt_for_llm = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{full_prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

req = json.dumps({
    'prompt': prompt_for_llm,
    'temperature': 1.0,
    'top_p': 1.0,
    'max_gen_len': 200
})
```


```python
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
response = bedrock_runtime.invoke_model(body=req, modelId='meta.llama3-70b-instruct-v1:0')
body = json.loads(response.get('body').read())
gen_text = body['generation']
print(gen_text)
```

    According to the context, the UK unemployment rate has increased recently, specifically from 4.4% to 4.5% in the January to March period.

