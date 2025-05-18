
from datetime import datetime
import json
import os
import subprocess
from typing import Any, Dict
import logging

import requests

logging.basicConfig(level=logging.INFO)

def ensure_data_path(path: str) -> str:
    """Ensure the path uses  '/data/...' """
    if path.startswith("/data/"):
        # if not running inside docker container, replace /data/ with ./data/
        if os.path.exists('/.dockerenv'):
            return path
        else:
            return f".{path}"
    raise ValueError("Path should start with '/data/'")

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

GENERATE_DATA_TOOL = {  
    "type": "function",
    "function": {
        "name": "generate_data",
        "description": "Generate data using uv",
        "parameters": {
            "type": "object",
            "properties": {
                "script_path": {
                    "type": "string",
                    "description": "full http path of the script to be used to generate the data"
                }
            },
            "required": ["script_path"],
            "additionalProperties": False
        },
        "strict": True,
    }
}

def generate_data(script_path: str):
    # log the script path
    logging.info(f"Script path: {script_path}")
    download_script(script_path, "datagen.py")
    # add email as argument to the script

    if os.path.exists('/.dockerenv'):
        subprocess.run(["uv", "run", "datagen.py", "24f1000472@ds.study.iitm.ac.in"])
    else:
        # process is running locally, create file in ./data directory
        subprocess.run(["python", "datagen.py", "--root", "./data", "24f1000472@ds.study.iitm.ac.in"])    

    logging.info("Data generated successfully")
    return "Data generated successfully"

# # implement function that will install a formatter of choice and version
# # using npm
# def install_formatter(formatter: str, version: str | None):
#     check_and_install_npm()
#     try:
#         # if version is not provided, install the latest version
#         if version is None or version == "latest":
#             subprocess.run(["npm", "install", formatter])
#         else:
#             subprocess.run(["npm", "install", f"{formatter}@{version}"])
#     except Exception as e:
#         raise ValueError(str(e))
#     return f"Formatter {formatter} installed successfully" 

# # implement function that checks if npm is installed and installs it if not
# def check_and_install_npm():
#     try:
#         subprocess.run(["npm", "--version"])
#     except FileNotFoundError:
#         subprocess.run(["sudo", "apt", "install", "npm"])
#         return "npm installed successfully"
#     except subprocess.CalledProcessError:
#         raise ValueError("Error occurred while checking npm installation")
#     except Exception as e:
#         raise ValueError(str(e))
#     return "npm already installed"

FORMAT_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "format_file",
        "description": "Format a file using a given formatter",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to format"
                },
                "formatter": {
                    "type": "string",
                    "description": "The formatter to use"
                },
                "version": {
                    "type": "string",
                    "description": "The version of the formatter to use"
                }
            },
            "required": ["path", "formatter", "version"],
            "additionalProperties": False
        },
        "strict": True,
    }
}

# implement function that will format a file of given path using a given formatter of choice
def format_file(path: str, formatter: str, version: str | None):
    # install_formatter(formatter, version)
    logging.info(f"Formatting file {path} using {formatter} version {version}")
    path = ensure_data_path(path)
    try:
        if version:
            subprocess.run(["npx", f"{formatter}@{version}", "--write", path])
        else:
            subprocess.run(["npx", f"{formatter}", "--write", path])
    except Exception as e:
        raise ValueError(str(e))
    logging.info(f"File formatted successfully")
    return "File formatted successfully"

# implement function that will read a file of given path
def read_file(path: str):
    logging.info(f"Reading file {path}")
    path = ensure_data_path(path)
    try:
        with open(path, "r") as file:
            content = file.read()
    except FileNotFoundError:
        raise ValueError("File not found")
    except Exception as e:
        raise ValueError(str(e))
    return content

COUNT_WEEKDAY_TOOL = {
    "type": "function",
    "function": {
        "name": "count_weekday_in_file",
        "description": "Count the number of weekdays in a file",
        "parameters": {
            "type": "object",
            "properties": {
                "in_path": {
                    "type": "string",
                    "description": "The file path to read"
                },
                "weekday": {
                    "type": "string",
                    "description": "The weekday to count"
                },
                "out_path": {
                    "type": "string",
                    "description": "The file path to write the count"
                }
            },
            "required": ["in_path", "weekday", "out_path"],
            "additionalProperties": False
        },
        "strict": True,
    }
}
# implement function that will reads lines from a file of given path, find date in that line, and finds out how many of those dates are of given weekday.
def count_weekday_in_file(in_path: str, weekday: str, out_path: str):
    logging.info(f"Counting {weekday} in file {in_path}")
    in_path = ensure_data_path(in_path)
    out_path = ensure_data_path(out_path)
    try:
        with open(in_path, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        raise ValueError("File not found")
    except Exception as e:
        raise ValueError(str(e))
    count = 0
    for line in lines:
        # get date from line
        date_str = line.strip()

        # check if date is of given weekday
        # if yes, increment count
        # we do not know the format of date, so we will try to parse it using different formats
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            try:
                date_obj = datetime.strptime(date_str, "%d-%m-%Y")
            except ValueError:
                try:
                    date_obj = datetime.strptime(date_str, "%m-%d-%Y")
                except ValueError:
                    continue        
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        if date_obj.strftime("%A") == weekday:
            count += 1
        
        # write the count to out_path
        with open(out_path, "w") as file:
            file.write(str(count))
    logging.info(f"{weekday} count written to {out_path}")
    return count

SORT_CONTACT_TOOL = {
    "type": "function",
    "function": {
        "name": "sort_contacts",
        "description": "Sort contacts by last name and first name",
        "parameters": {
            "type": "object",
            "properties": {
                "in_path": {
                    "type": "string",
                    "description": "The file path to read contacts"
                },
                "out_path": {
                    "type": "string",
                    "description": "The file path to write sorted contacts"
                }
            },
            "required": ["in_path", "out_path"],
            "additionalProperties": False
        },
        "strict": True,
    }
}
# implement function that reads an array of contacts in json objects from a file of given path, 
# sorts them by last_name and then by first_name, 
# and writes them back to another file of given path
def sort_contacts(in_path: str, out_path: str):
    logging.info(f"Sorting contacts in file {in_path}")
    in_path = ensure_data_path(in_path)
    out_path = ensure_data_path(out_path)
    try:
        with open(in_path, "r") as file:
            contacts = json.load(file)
    except FileNotFoundError:
        raise ValueError("File not found")
    except Exception as e:
        raise ValueError(str(e))
    contacts.sort(key=lambda x: (x["last_name"], x["first_name"]))
    with open(out_path, "w") as file:
        json.dump(contacts, file)
    logging.info(f"Contacts sorted and written to {out_path}")
    return "Contacts sorted successfully"

READ_RECENT_FILES_TOOL = {
    "type": "function",
    "function": {
        "name": "read_recent_files_of_type",
        "description": "Read first line of top 10 recent files of given extension",
        "parameters": {
            "type": "object",
            "properties": {
                "extension": {
                    "type": "string",
                    "description": "The file extension to filter files"
                },
                "dir_path": {
                    "type": "string",
                    "description": "The directory path to read files"
                },
                "out_path": {
                    "type": "string",
                    "description": "The file path to write first lines"
                }
            },
            "required": ["extension", "dir_path", "out_path"],
            "additionalProperties": False
        },
        "strict": True,
    }
}
# implement function that 
# reads files of given extension from a directory of given path,
# sorts them in reverse by the file name
# and writes their first line of top 10 files to another file of given path
def read_recent_files_of_type(extension: str, dir_path: str, out_path: str):
    dir_path = ensure_data_path(dir_path)
    out_path = ensure_data_path(out_path)

    try:
        files = os.listdir(dir_path)
    except FileNotFoundError:
        raise ValueError("Directory not found")
    except Exception as e:
        raise ValueError(str(e))
    # filter files by extension
    files = [file for file in files if file.endswith(extension)]
    # sort files by their file name in reverse
    files.sort(key=lambda x: x, reverse=True)
    with open(out_path, "w") as out_file:
        for file in files[:10]:
            with open(os.path.join(dir_path, file), "r") as f:
                out_file.write(f.readline())
    return "First lines of recent files written successfully"

READ_MD_HEADINGS_TOOL = {
    "type": "function",
    "function": {
        "name": "read_md_headings",
        "description": "Read first heading one line of all markdown files in a directory",
        "parameters": {
            "type": "object",
            "properties": {
                "dir_path": {
                    "type": "string",
                    "description": "The directory path to read markdown files"
                },
                "out_path": {
                    "type": "string",
                    "description": "The file path to write headings"
                }
            },
            "required": ["dir_path", "out_path"],
            "additionalProperties": False
        },
        "strict": True,
    }
}
# implement function that will do following:
# - read all files of extension *.md from a directory of given path,
# - look for first line that starts with # . This is a heading one line
# - create an dictionary with filename with path relative from given path as key and first heading one as value
# - write this dictionary as json to a output file of given path
def read_md_headings(dir_path: str, out_path: str):
    dir_path = ensure_data_path(dir_path)
    out_path = ensure_data_path(out_path)
    try:
        files = os.listdir(dir_path)
    except FileNotFoundError:
        raise ValueError("Directory not found")
    except Exception as e:
        raise ValueError(str(e))
    # filter files by extension
    files = [file for file in files if file.endswith(".md")]
    headings = {}
    for file in files:
        with open(os.path.join(dir_path, file), "r") as f:
            for line in f:
                if line.startswith("#"):
                    # value should not contain leading #, so strip it
                    headings[os.path.relpath(os.path.join(dir_path, file))] = line.strip('#').strip()
                    break
    with open(out_path, "w") as file:
        json.dump(headings, file)
    return "Headings read successfully"

EXTRACT_SENDER_EMAIL_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_sender_email",
        "description": "Extract sender's email address from an email text",
        "parameters": {
            "type": "object",
            "properties": {
                "in_path": {
                    "type": "string",
                    "description": "The file path to read email text"
                },
                "out_path": {
                    "type": "string",
                    "description": "The file path to write sender's email"
                }
            },
            "required": ["in_path", "out_path"],
            "additionalProperties": False
        },
        "strict": True,
    }
}
# implement function that will do following:
# - read a text of an email from given file path
# - uses LLM to find out sender's email address
# - writes that email address to another file of given path
def extract_sender_email(in_path: str, out_path: str):
    in_path = ensure_data_path(in_path)
    out_path = ensure_data_path(out_path)
    try:
        with open(in_path, "r") as in_file:
            text = in_file.read()

            email = gpt_query_content(f"Extract the sender's email address from the following email text:\n\n{text}")
            with open(out_path, "w") as out_file:
                out_file.write(email)
                return f"Sender's email {email} extracted successfully"   

    except FileNotFoundError:
        raise ValueError("File not found")
    except Exception as e:
        raise ValueError(str(e))


AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
API_BASE = "https://aiproxy.sanand.workers.dev/openai/v1"
LLM_MODEL = "gpt-4o-mini"

def gpt_query_content(prompt: str) -> str:
    logging.info(f"AIPROXY_TOKEN: {AIPROXY_TOKEN}")
    logging.info(f"API_BASE: {API_BASE}")
    logging.info(f"LLM_MODEL: {LLM_MODEL}")
    logging.info(f"Prompt: {prompt}")

    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        raise Exception(f"LLM API error: {response.text}")

    return response.json()["choices"][0]["message"]["content"]



def gpt_query_function_call(user_input: str, tools: list[Dict[str, Any]]) -> str:
    # log aiproxy token, api base and llm model
    logging.info(f"AIPROXY_TOKEN: {AIPROXY_TOKEN}")
    logging.info(f"API_BASE: {API_BASE}")
    logging.info(f"LLM_MODEL: {LLM_MODEL}")
    # log the user input and tools
    logging.info(f"User input: {user_input}")
    logging.info(f"Tools: {tools}")

    url = f"{API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": user_input}],
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 100
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code != 200:
        raise Exception(f"LLM API error: {response.text}")

    message = response.json()["choices"][0]["message"]

    function_calls = []
    for tool_call in message["tool_calls"]:
        function_call = {}
        function_call["name"] = tool_call["function"]["name"]
        function_call["arguments"] = json.loads( tool_call["function"]["arguments"])
        function_calls.append(function_call)
    logging.info(function_calls)
    return function_calls

def find_and_run_task(user_input: str) -> str:
    tools = [
        GENERATE_DATA_TOOL,
        FORMAT_FILE_TOOL,
        COUNT_WEEKDAY_TOOL,
        SORT_CONTACT_TOOL,
        READ_RECENT_FILES_TOOL,
        READ_MD_HEADINGS_TOOL,
        EXTRACT_SENDER_EMAIL_TOOL
    ]
    function_calls = gpt_query_function_call(user_input, tools)
    # log the function calls
    logging.info(f"Function calls: {function_calls}")
    for function_call in function_calls:
        function_name = function_call["name"]
        arguments = function_call["arguments"]
        if function_name == "generate_data":
            generate_data(**arguments)
            logging.info("Data generated successfully")
            return "Data generated successfully"
        elif function_name == "format_file":
            format_file(**arguments)
            logging.info("File formatted successfully")
            return "File formatted successfully"
        elif function_name == "count_weekday_in_file":
            count_weekday_in_file(**arguments)
            logging.info("Weekday count written successfully")
            return "Weekday count written successfully"
        elif function_name == "sort_contacts":
            sort_contacts(**arguments)
            logging.info("Contacts sorted successfully")
            return "Contacts sorted successfully"
        elif function_name == "read_recent_files_of_type":
            read_recent_files_of_type(**arguments)
            logging.info("First lines of recent files written successfully")
            return "First lines of recent files written successfully"
        elif function_name == "read_md_headings":
            read_md_headings(**arguments)
            logging.info("Headings read successfully")
            return "Headings read successfully"
        elif function_name == "extract_sender_email":
            extract_sender_email(**arguments)
            logging.info("Sender's email extracted successfully")
            return "Sender's email extracted successfully"
    logging.error("Could not find a suitable function to run")
    return "Could not find a suitable function to run"
