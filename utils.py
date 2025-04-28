import json
from typing import Union, List, Dict, Any
import pandas as pd
from graphrag.query.structured_search.base import SearchResult

def convert_response_to_string(response: Union[str, Dict[str, Any], List[Dict[str, Any]]]) -> str:
    """
    Convert a response that can be a string, dictionary, or list of dictionaries to a string.
    """
    if isinstance(response, (dict, list)):
        return json.dumps(response)
    elif isinstance(response, str):
        return response
    else:
        return str(response)

def recursively_convert(obj: Any) -> Any:
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, list):
        return [recursively_convert(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: recursively_convert(value) for key, value in obj.items()}
    return obj

def process_context_data(context_data: Union[str, List[pd.DataFrame], Dict, pd.DataFrame]) -> Any:
    if isinstance(context_data, str):
        return context_data
    if isinstance(context_data, pd.DataFrame):
        return context_data.to_dict(orient="records")
    if isinstance(context_data, (list, dict)):
        return recursively_convert(context_data)
    return None

def serialize_search_result(search_result: SearchResult) -> Dict[str, Any]:
    return {
        "response": search_result.response,
        "context_data": process_context_data(search_result.context_data),
        "context_text": search_result.context_text,
        "completion_time": search_result.completion_time,
        "llm_calls": search_result.llm_calls,
        "prompt_tokens": search_result.prompt_tokens
    }