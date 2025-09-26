import json
import pandas as pd
from io import StringIO
from pprint import pprint
import rich


result_file_name = "batch_68cf1fac07388190a0b09131fe2db69b_output.jsonl"


# Loading data from saved file
results = []
with open(result_file_name, 'r') as file:
    for line in file:
        # Parsing the JSON string into a dict and appending to the list of results
        json_object = json.loads(line.strip())
        results.append(json_object)




# Assuming 'results' is a list of response dictionaries
data_list = []

for item in results:
    try:
        embedding = item['response']['body']['data'][0]['embedding']
        custom_id = item['custom_id']

        data_list.append({
            'custom_id': custom_id,
            'embedding': embedding
        })
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Skipping item due to error: {e}")



# Create DataFrame
df = pd.DataFrame(data_list)
rich.print(df.head())
df.to_csv("book_embeddings.csv", index=False)

