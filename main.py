import json
import logging
import os
import traceback
from typing import Callable, Dict
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import requests

from function_tasks import (
    clone_git_repo_and_commit,
    compress_image,
    convert_function_to_openai_schema,
    convert_markdown_to_html,
    fetch_data_from_api_and_save,
    filter_csv,
    format_file_with_prettier,
    query_database, 
    extract_specific_text_using_llm, 
    get_similar_text_using_embeddings, 
    extract_text_from_image, 
    extract_specific_content_and_create_index, 
    process_and_write_logfiles,
    run_sql_query_on_database,
    scrape_webpage, 
    sort_json_by_keys, 
    count_occurrences, 
    install_and_run_script,
    transcribe_audio
)

logging.basicConfig(level=logging.INFO)

app = FastAPI()

    
def ensure_local_path(path: str) -> str:
    """Ensure the path uses  '/data/...' """
    if path.startswith("/data/"):
        # if not running inside docker container, replace /data/ with ./data/
        if os.path.exists('/.dockerenv'):
            return path
        else:
            return path.lstrip("/")
    raise ValueError("Path should start with '/data/'")


function_mappings: Dict[str, Callable] = {
    "install_and_run_script": install_and_run_script, 
    "format_file_with_prettier": format_file_with_prettier,
    "query_database": query_database, 
    "extract_specific_text_using_llm": extract_specific_text_using_llm, 
    'get_similar_text_using_embeddings': get_similar_text_using_embeddings, 
    'extract_text_from_image': extract_text_from_image, 
    "extract_specific_content_and_create_index": extract_specific_content_and_create_index, 
    "process_and_write_logfiles": process_and_write_logfiles, 
    "sort_json_by_keys": sort_json_by_keys, 
    "count_occurrences": count_occurrences,
    # "fetch_data_from_api_and_save": fetch_data_from_api_and_save,
    "clone_git_repo_and_commit": clone_git_repo_and_commit,
    "run_sql_query_on_database": run_sql_query_on_database,
    "scrape_webpage": scrape_webpage,
    "compress_image": compress_image,
    "transcribe_audio": transcribe_audio,
    "convert_markdown_to_html": convert_markdown_to_html,
    "filter_csv": filter_csv,

}

URL_CHAT = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
API_KEY = os.getenv("AIPROXY_TOKEN")

def parse_task_description(task_description: str, tools: list):
    response = requests.post(
        URL_CHAT,
        headers={"Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{
                'role': 'system',
                'content': "You are intelligent agent that understands and parses tasks. You quickly identify the best tool functions to use to give the desired results"
            },
            {
                "role": "user",
                "content": task_description
            }],
            "tools": tools,
            "tool_choice": "required",
        }
    )
    logging.info("PRINTING RESPONSE:::" * 3)
    print(response.json())
    logging.info("PRINTING RESPONSE:::" * 3)
    return response.json()["choices"][0]["message"]

def execute_function_call(function_call):
    logging.info(f"Inside execute_function_call with function_call: {function_call}")
    try:
        function_name = function_call["name"]
        function_args = json.loads(function_call["arguments"])
        function_to_call = function_mappings.get(function_name)
        
        logging.info("PRINTING RESPONSE:::" * 3)
        print('Calling function:', function_name)
        print('Arguments:', function_args)
        logging.info("PRINTING RESPONSE:::" * 3)
        
        if function_to_call:
            function_to_call(**function_args)
        else:
            raise ValueError(f"Function {function_name} not found")
    except Exception as e:
        error_details = traceback.format_exc()
        logging.error(f"Error in execute_function_call: {error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"Error executing function: {str(e)}"
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail)}
    )


@app.post("/run")
async def run_task(task: str = Query(..., description="Plain-English task description")):
    tools = [convert_function_to_openai_schema(func) for func in function_mappings.values()]
    logging.info(len(tools))
    logging.info(f"Inside run_task with task: {task}")
    try:
        function_call_response_message = parse_task_description(task, tools)
        if function_call_response_message["tool_calls"]:
            for tool in function_call_response_message["tool_calls"]:
                execute_function_call(tool["function"])
        return {"status": "success", "message": "Task executed successfully"}
    except Exception as e:
        error_details = traceback.format_exc()
        logging.error(f"Error in run_task: {error_details}")
        raise HTTPException(
            status_code=500,
            detail=f"Error executing task: {str(e)}"
        )
    
@app.get("/read", response_class=PlainTextResponse)
async def read_file(path: str = Query(..., description="Path to the file to read")):
    logging.info(f"Inside read_file with path: {path}")
    output_file_path = ensure_local_path(path)
    if not os.path.exists(output_file_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        with open(output_file_path, "r") as file:
            content = file.read()
        return content
    except Exception as e:
        logging.error(f"Error reading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
