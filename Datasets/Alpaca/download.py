import os
import json
import urllib
from urllib import request

def download_and_load_file(file_path, url):

    if not os.path.exists(file_path):
        with request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    with open(file_path, "r") as file:
        data = json.load(file)

    return data

if __name__ == "__main__":

    file_path = "instruction-data-alpaca.json"
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    data = download_and_load_file(file_path, url)
