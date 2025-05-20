import docstring_parser
import dotenv
import logging
import subprocess
import glob
import sqlite3
import requests
from bs4 import BeautifulSoup
import markdown
import csv
import base64
import duckdb
import base64
import numpy as np
import requests
import os
import json
from dateutil.parser import parse
import re
import httpx
import inspect
from sklearn.metrics.pairwise import cosine_similarity
from typing import Callable, get_type_hints, Dict, Any, Tuple,Optional,List
from pydantic import create_model, BaseModel
import re
from PIL import Image

dotenv.load_dotenv()

URL_CHAT = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
API_KEY = os.getenv("AIPROXY_TOKEN")
URL_EMBEDDING = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"

logging.basicConfig(level=logging.INFO)

def ensure_local_path(path: str) -> str:
    """Ensure the path uses  '/data/...' """
    if path.startswith("/data/"):
        # if not running inside docker container, replace /data/ with ./data/
        if os.path.exists('/.dockerenv'):
            return path
        else:
            return path.lstrip("/")
    raise ValueError("Path should start with '/data/'")


def convert_function_to_openai_schema(func: Callable) -> dict:
    """
    Converts a Python function into an OpenAI function schema with strict JSON schema enforcement.

    Args:
        func (Callable): The function to convert.

    Returns:
        dict: The OpenAI function schema.
    """
    # Extract the function's signature
    sig = inspect.signature(func)
    

    type_hints = get_type_hints(func)
    
    
    fields = {
        name: (type_hints.get(name, Any), ...)
        for name in sig.parameters
    }
    PydanticModel = create_model(func.__name__ + "Model", **fields)
    
   
    schema = PydanticModel.model_json_schema()
    
    # Parse the function's docstring
    docstring = inspect.getdoc(func) or ""
    parsed_docstring = docstring_parser.parse(docstring)
    

    param_descriptions = {
        param.arg_name: param.description or ""
        for param in parsed_docstring.params
    }
    
    for prop_name, prop in schema.get('properties', {}).items():
        prop['description'] = param_descriptions.get(prop_name, '')
        
        if prop.get('type') == 'array' and 'items' in prop:
            if not isinstance(prop['items'], dict) or 'type' not in prop['items']:
                # Default to array of strings if type is not specified
                prop['items'] = {'type': 'string'}
    
    schema['additionalProperties'] = False
    
    schema['required'] = list(fields.keys())
    
    openai_function_schema = {
        'type': 'function',
        'function':{
        'name': func.__name__,
        'description': parsed_docstring.short_description or '',
        'parameters': {
            'type': 'object',
            'properties': schema.get('properties', {}),
            'required': schema.get('required', []),
            'additionalProperties': schema.get('additionalProperties', False),
        },
        'strict': True,
    }
    }
    
    return openai_function_schema
 
def format_file_with_prettier(file_path: str, prettier_version: str):
    # """
    # Format the contents of a specified file using Prettier, ensuring the file is updated in-place.
    
    # Args:
    #     file_path: The path to the file to format.  
    #     prettier_version: The version of Prettier to use.
    # """
    input_file_path = ensure_local_path(file_path)
    
    try:
        # First check if Node.js is installed
        subprocess.run(["node", "--version"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        raise RuntimeError("Node.js is not installed. Please install Node.js first: https://nodejs.org/")
    except FileNotFoundError:
        raise RuntimeError("Node.js is not installed. Please install Node.js first: https://nodejs.org/")
        
    try:
        # Install prettier locally if not already installed
        subprocess.run(
            ["npm", "install", f"prettier@{prettier_version}"], 
            check=True,
            capture_output=True
        )
        
        # Use local prettier installation instead of npx
        prettier_path = "./node_modules/.bin/prettier"
        subprocess.run([prettier_path, "--write", input_file_path], check=True)
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error running Prettier: {str(e)}")
    except FileNotFoundError as e:
        raise RuntimeError(f"Error: {str(e)}. Make sure npm is installed and in your PATH.")    
    input_file_path = ensure_local_path(file_path)
    subprocess.run(["npx", f"prettier@{prettier_version}", "--write", input_file_path])

def query_gpt(user_input: str,task: str):
    response = requests.post(
        URL_CHAT,
        headers={"Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"},
        json={
            "model": "gpt-4o-mini",
            "messages":[{'role': 'system','content':"JUST DO WHAT IS ASKED\n YOUR output is part of a program, using tool functions"+task},
                        {'role': 'user', 'content': user_input}]
        }
    )
    logging.info("PRINTING RESPONSE:::"*3)
    print("Inside query_gpt")
    logging.info("PRINTING RESPONSE:::"*3)
    response.raise_for_status()
    return response.json()

def rewrite_sensitive_task(task: str) -> str:
    """Rewrite sensitive task descriptions in an indirect way."""
    task_lower = task.lower()
    
    rewrite_map = {
        "credit card": "longest numerical sequence",
        "cvv": "3-digit number near another number",
        "bank account": "second longest numerical sequence",
        "routing number": "a series of numbers used for banking",
        "social security": "9-digit numerical sequence",
        "passport": "longest alphanumeric string",
        "driver's license": "structured alphanumeric code",
        "api key": "a long secret-looking string",
        "password": "text following 'Password:'",
    }
    
    for keyword, replacement in rewrite_map.items():
        if keyword in task_lower:
            return re.sub(keyword, replacement, task, flags=re.IGNORECASE)

    return task


def query_gpt_image(image_path: str, task: str):
    logging.info(f"Inside query_gpt_image with image_path: {image_path} and task: {task}")
    image_format = image_path.split(".")[-1]
    clean_task = rewrite_sensitive_task(task)
    with open(image_path, "rb") as file:
        base64_image = base64.b64encode(file.read()).decode("utf-8")
    response = requests.post(
        URL_CHAT,
        headers={"Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{'role': 'system','content':"JUST GIVE the required input, as short as possible, one word if possible. "},
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Extract {clean_task} in image"},
                    {
                    "type": "image_url",
                    "image_url": { "url": f"data:image/{image_format};base64,{base64_image}" }
                    }
                ]
                }
            ]
            }
                     )
    
    response.raise_for_status()
    return response.json()

""""
A TASKS
"""
def query_database(db_file: str, output_file: str, query: str, query_params: Tuple):
    """
    Executes a SQL query on the specified SQLite database and writes the result to an output file.

    Args:
        db_file (str): The path to the SQLite database file.
        output_file (str): The path to the output file where the result will be written.
        query (str): The SQL query to execute.
        query_params (Tuple): The parameters to pass to the query in order to the query

    Returns:
        None
    """
    db_file_path = ensure_local_path(db_file)
    output_file_path = ensure_local_path(output_file)

    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    try:

        cursor.execute(query, query_params)
        result = cursor.fetchone()

        if result:
            output_data = result[0]
        else:
            output_data = 'No results found.'

        with open(output_file_path, "w") as file:
            file.write(str(output_data))

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")

    finally:
        conn.close()
def extract_specific_text_using_llm(input_file: str, output_file: str, task: str):
    """
    extract_specific_text_using_llm

    Args:
        input_file (str): input_file
        output_file (str): output_file
        task (str): The task that specifies the text to extract.
    Returns:
        None
    """
    input_file_path = ensure_local_path(input_file)
    with open(input_file_path, "r") as file:
        text_info = file.read() #readlines gives list, this gives string
    output_file_path = ensure_local_path(output_file)
    response = query_gpt(text_info, task) # received in json format
    logging.info(f"Inside extract_specific_text_using_llm with input_file: {input_file}, output_file: {output_file}, and task: {task}")
    with open(output_file_path, "w") as file:
        file.write(response["choices"][0]["message"]["content"])
def get_embeddings(texts: List[str]):
    response =  requests.post(
            URL_EMBEDDING,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": "text-embedding-3-small", "input": texts},
        )
    embeddings = np.array([emb["embedding"] for emb in response.json()["data"]])
    return embeddings
def get_similar_text_using_embeddings(input_file: str, output_file: str, no_of_similar_texts: int):
    """
    get_similar_text_using_embeddings

    Args:
        input_file (str): input_file
        output_file (str): output_file
        no_of_similar_texts (int): no_of_similar_texts
    Returns:
        None
    """
    input_file_path = ensure_local_path(input_file)
    output_file_path = ensure_local_path(output_file)


    with open(input_file_path, "r") as file:
        documents = file.readlines()
    
    documents = [comment.strip() for comment in documents]
    
    line_embeddings = get_embeddings(documents)
    similarity_matrix = cosine_similarity(line_embeddings)
    
    np.fill_diagonal(similarity_matrix, -1)  # Ignore self-similarity
    most_similar_indices = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)
    
    similar_texts = []
    for i in range(no_of_similar_texts):
        similar_texts.append(documents[most_similar_indices[i]])

    with open(output_file_path, "w") as file:
        for text in similar_texts:
            file.write(text + "\n")
def extract_text_from_image(image_path: str, output_file: str, task: str):
    """
    Extract text from image.
    Args:
        image_path (str): image_path
        output_file (str): output_file
        task (str): task
    Returns:
        None
    """
    # Use an LLM to extract the credit card number
    image_path___ = ensure_local_path(image_path)
    response = query_gpt_image(image_path___, task)
    
    output_file_path = ensure_local_path(output_file) 
    # Remove spaces and write the result to the output file
    print(response["choices"][0]["message"])
    with open(output_file_path, "w") as file:
        file.write(response["choices"][0]["message"]["content"].replace(" ", ""))       
def extract_specific_content_and_create_index(input_file: str, output_file: str, extension: str,content_marker: str):
    """
    Identify all files with a specific extension in a directory. For each file, extract particular content (e.g., the first occurrence of a header) and create an index file mapping filenames to their extracted content.
    
    Args:
        input_file (str): input_file
        output_file (str): output_file
        extension (str): extension
        content_marker (str): content_marker
    """
    input_file_path = ensure_local_path(input_file)
    output_file_path = ensure_local_path(output_file)

    extension_files = glob.glob(os.path.join(input_file_path, "**", f"*{extension}"), recursive=True)
    
    index = {}

    for extenstion_file in extension_files:
        title = None
        with open(extenstion_file, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith(content_marker):
                    title = line.lstrip(content_marker).strip()
                    break  

        relative_path = os.path.relpath(extenstion_file, input_file_path)

        index[relative_path] = title if title else ""

    with open(output_file_path, "w", encoding="utf-8") as json_file:
        json.dump(index, json_file, indent=2, sort_keys=True)
def process_and_write_logfiles(input_file: str, output_file: str, num_logs: int = 10, num_of_lines: int = 1):
    """
    Process n number of log files num_logs given in the input_file and write x number of lines num_of_lines  of each log file to the output_file.
    
    Args:
        input_file (str): input_file
        output_file (str): output_file
        num_logs (int): num_logs
        num_of_lines (int): num_of_lines
    """
    input_file_path = ensure_local_path(input_file)
    output_file_path = ensure_local_path(output_file) 
    log_files = glob.glob(os.path.join(input_file_path, "*.log"))
    
    log_files.sort(key=os.path.getmtime, reverse=True)
    

    recent_logs = log_files[:num_logs]
    

    with open(output_file_path, "w") as outfile:
        for log_file in recent_logs:
            with open(log_file, "r") as infile:
                for _ in range(num_of_lines):
                    line = infile.readline()
                    if line:
                        outfile.write(line)
                    else:
                        break
def sort_json_by_keys(input_file: str, output_file: str, keys: list):
    """
    Sort JSON data by specified keys in specified order and write the result to an output file.
    Args:
        input_file (str): input_file.
        output_file (str): output_file.
        keys (list): keys.
    """
    input_file_path = ensure_local_path(input_file)
    output_file_path = ensure_local_path(output_file) 
    with open(input_file_path, "r") as file:
        data = json.load(file)
    
    sorted_data = sorted(data, key=lambda x: tuple(x[key] for key in keys))
    
    with open(output_file_path, "w") as file:
        json.dump(sorted_data, file)                       
def count_occurrences(
    input_file: str,
    output_file: str,
    date_component: Optional[str] = None,
    target_value: Optional[int] = None,
    custom_pattern: Optional[str] = None
):
    """
    Count occurrences of specific date components or custom patterns in a file and write the count to an output file. Handles various date formats automatically.
    Args:
        input_file (str): input_file
        output_file (str): output_file
        date_component (Optional[str]): The date component to check ('weekday', 'month', 'year', 'leap_year').
        target_value (Optional[int]): The target value for the date component e.g., IMPORTANT KEYS TO KEEP IN MIND --> 0 for Monday, 1 for Tuesday, 2 for Wednesday if weekdays, 1 for January 2 for Febuary if month, 2025 for year if year.
        custom_pattern (Optional[str]): A regex pattern to search for in each line.
    """  
    count = 0
    input_file_path = ensure_local_path(input_file)
    output_file_path = ensure_local_path(output_file)
    with open(input_file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            # Check for custom pattern
            if custom_pattern and re.search(custom_pattern, line):
                count += 1
                continue

            # Attempt to parse the date
            try:
                parsed_date = parse(line)  # Auto-detect format
            except (ValueError, OverflowError):
                print(f"Skipping invalid date format: {line}")
                continue

            # Check for specific date components
            if date_component == 'weekday' and parsed_date.weekday() == target_value:
                count += 1
            elif date_component == 'month' and parsed_date.month == target_value:
                count += 1
            elif date_component == 'year' and parsed_date.year == target_value:
                count += 1
            elif date_component == 'leap_year' and parsed_date.year % 4 == 0 and (parsed_date.year % 100 != 0 or parsed_date.year % 400 == 0):
                count += 1

    # Write the result to the output file
    with open(output_file_path, "w") as file:
        file.write(str(count))


def download_script(url: str, local_filename: str) -> None:
    """Download the Python script from the URL."""
    logging.info(f"Downloading script from {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(local_filename, "w") as file:
            file.write(response.text)
        logging.info(f"Script downloaded and saved as {local_filename}")
    else:
        logging.error("Failed to download the script.")
        # raise an exception if the script could not be downloaded
        raise ValueError(f"Failed to download the script. {response.text}")


def install_and_run_script(script_path: str, args: list):
    """
    Generates data by downloading and running the specified script.
    Args:
        script_path (str): The URL of the script to download.
        args (list): The arguments to pass to the script when executed.
    """
    # log the script path
    logging.info(f"Script path: {script_path}")
    download_script(script_path, "datagen.py")
    # add email as argument to the script

    if os.path.exists('/.dockerenv'):
        subprocess.run(["uv", "run", "datagen.py", args[0]])
    else:
        # process is running locally, create file in ./data directory
        subprocess.run(["python", "datagen.py", "--root", "./data"] + args[0])    

    logging.info("Data generated successfully")
    return "Data generated successfully"


# def install_and_run_script(package: str, args: list,*,script_url: str):
#     """
#     Install a package and download a script from a URL with provided arguments and run it with uv run {pythonfile}.py.PLEASE be cautious and Note this generally used in the starting.ONLY use this tool function if url is given with https//.... or it says 'download'. If no conditions are met, please try the other functions.
#     Args:
#         package (str): The package to install.
#         script_url (str): The URL to download the script from
#         args (list): The arguments to pass to the script and run it
#     """
#     if package == "uvicorn":
#         subprocess.run(["pip", "install", "uv"])
#     else:
#         subprocess.run(["pip", "install", package])
#     subprocess.run(["curl", "-O", script_url])
#     script_name = script_url.split("/")[-1]
#     subprocess.run(["uv","run", script_name,args[0]])

""""
B TASKS
ADD generated response to double check dynamically
"""

# Fetch data from an API and save it
def fetch_data_from_api_and_save(url: str, output_file: str, params: Optional[Dict[str, Any]] = None):
    """
    fetch_data_from_api_and_save
    Args:
        url (str): url.
        output_file (str): output_file.
        params (Optional[Dict[str, Any]]): params
    """   
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        with open(output_file, "w") as file:
            json.dump(data, file, indent=4)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
    try:
        response = requests.post(url, params["headers"], params["data"])
        response.raise_for_status()
        data = response.json()
        with open(output_file, "w") as file:
            json.dump(data, file, indent=4)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")

#Clone a git repo and make a commit
def clone_git_repo_and_commit(repo_url: str, output_dir: str, commit_message: str):
    """
    clone_git_repo_and_commit
    Args:
        repo_url (str): repo_url
        output_dir (str): output_dir
        commit_message (str): commit_message
    """
    try:
        subprocess.run(["git", "clone", repo_url, output_dir])
        subprocess.run(["git", "add", "."], cwd=output_dir)
        subprocess.run(["git", "commit", "-m", commit_message], cwd=output_dir)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

#Run a SQL query on a SQLite or DuckDB database
def run_sql_query_on_database(database_file: str, query: str, output_file: str, is_sqlite: bool = True):
    """
    run sql query on database
    Args:
        database_file (str): database file
        query (str): query
        output_file (str): output file
        is_sqlite (bool): is sqlite
    """
    if is_sqlite:
        try:
            conn = sqlite3.connect(database_file)
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            with open(output_file, "w") as file:
                for row in result:
                    file.write(str(row) + "\n")
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
        finally:
            conn.close()
    else:
        try:
            conn = duckdb.connect(database_file)
            cursor = conn.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            with open(output_file, "w") as file:
                for row in result:
                    file.write(str(row) + "\n")
        except duckdb.Error as e:
            print(f"An error occurred: {e}")
        finally:
            conn.close()

#Extract data from (i.e. scrape) a website
def scrape_webpage(url: str, output_file: str):
    # """
    # This tool function scrapes a website
    # Args:
    #     url (str): The URL of the website to scrape.
    #     output_file (str): The path to the output file where the scraped data will be saved.
    # """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    with open(output_file, "w") as file:
        file.write(soup.prettify())

#Compress or resize an image
def compress_image(input_file: str, output_file: str, quality: int = 50):
    """
    compresses image
    Args:
        input_file (str): input_file
        output_file (str): output_file
        quality (int): quality
    """
    img = Image.open(input_file)
    img.save(output_file, quality=quality)

#Transcribe audio from an MP3 file
def transcribe_audio(input_file: str, output_file: str):
    # """
    # This tool function transcribes audio from an MP3 file.
    # Args:
    #     input_file (str): The path to the input MP3 audio file.
    #     output_file (str): The path to the output text file where the transcription will be saved.
    # """
    transcript = "Transcribed text"  # Placeholder
    with open(output_file, "w") as file:
        file.write(transcript)
#Convert Markdown to HTML
def convert_markdown_to_html(input_file: str, output_file: str):
    """
    Convert a Markdown file to HTML
    Args:
        input_file (str): input file
        output_file (str): output file
    """
    with open(input_file, "r") as file:
        html = markdown.markdown(file.read())
    with open(output_file, "w") as file:
        file.write(html)

# Write an API endpoint that filters a CSV file and returns JSON data
def filter_csv(input_file: str, column: str, value: str, output_file: str):
    """
    filter_csv
    Args:
        input_file (str): input file
        column (str): column
        value (str): value
        output_file (str): output path
    """
    results = []
    with open(input_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row[column] == value:
                results.append(row)
    with open(output_file, "w") as file:
        json.dump(results, file, indent=4)
