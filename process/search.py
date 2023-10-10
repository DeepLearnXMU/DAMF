from typing import Dict
import json
import requests
import urllib
import logging

logger = logging.getLogger()

SOGOU_URL = "xxxx"
SOGOU_PARA = "yyyy"
TIMEOUT = 10

def call_sogou_search(url: str, query: str, opts: Dict = None, timeout: float = TIMEOUT) -> Dict:
    """
    call sogou search
    """
    encode_query = urllib.parse.quote_plus(query)
    keyword = f"?keyword={encode_query}"
    site = ""
    if opts is not None:
        site = opts.get("site", "")
    if site.strip() != "":
        site = f"site:{site}"
    try:
        response = requests.get(f"{url}{keyword}{site}{SOGOU_PARA}", timeout=timeout)
        response.encoding = "utf-8"
        json_output = response.json()
        mock_result_dump = {
            "type": "mock_sogou",
            "query": query,
            "result": json_output,
        }
        logger.debug(json.dumps(mock_result_dump, ensure_ascii=False))
    except Exception as e:  # pylint: disable=broad-except
        # logger.error(f"Cannot parse sogou result! {e}")
        json_output = {}
    # logger.debug(f"Sogou raw output: {json_output}")
    return json_output