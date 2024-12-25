import os
import requests
import project_paths as pp

if 'words.txt' not in os.listdir(pp.datasets_folder_path):
    word_site = 'https://www.mit.edu/~ecprice/wordlist.10000'

    response = requests.get(word_site)
    words = response.content.splitlines()
    with open(os.path.join(pp.datasets_folder_path, 'words.txt'), 'w') as file:
        for word in words:
            file.write(f'{word.decode("UTF-8")}\n')