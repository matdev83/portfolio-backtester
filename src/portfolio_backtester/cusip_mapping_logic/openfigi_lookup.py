import json
import time
import logging
from typing import Optional, Tuple
import requests

logger = logging.getLogger(__name__)

def lookup_openfigi(api_key: str, ticker: str, *, throttle: float = 1.0) -> Tuple[Optional[str], Optional[str]]:
    if not api_key:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("OPENFIGI_API_KEY missing – skipping lookup")
        return None, None

    url = "https://api.openfigi.com/v3/mapping"
    payloads = [
        [{"idType": "TICKER", "idValue": ticker, "exchCode": "US"}],
        [{"idType": "TICKER", "idValue": ticker}]
    ]

    for payload in payloads:
        try:
            response_data = _query_openfigi_api(api_key, url, payload)
            if response_data and isinstance(response_data, list) and response_data[0].get("data"):
                for security_info in response_data[0]["data"]:
                    cusip = security_info.get("cusip")
                    name = security_info.get("securityDescription") or \
                           security_info.get("name") or \
                           security_info.get("securityName")

                    if cusip and 8 <= len(cusip) <= 9:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"OpenFIGI found CUSIP {cusip} for ticker {ticker}")
                        time.sleep(throttle)
                        return cusip, name
            elif response_data and isinstance(response_data, list) and "warning" in response_data[0]:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"OpenFIGI warning for {ticker} with payload {payload}: {response_data[0]['warning']}")

        except requests.RequestException as e:
            logger.warning(f"OpenFIGI API request failed for {ticker} with payload {payload}: {e}")
        except Exception as e:
            logger.error(f"Error processing OpenFIGI response for {ticker} with payload {payload}: {e}")

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"CUSIP not found via OpenFIGI for ticker {ticker}")
    return None, None

def _query_openfigi_api(api_key: str, url: str, payload: list) -> Optional[list]:
    if not api_key:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("OPENFIGI_API_KEY missing – skipping API query.")
        return None

    headers = {
        "Content-Type": "application/json",
        "X-OPENFIGI-APIKEY": api_key,
    }
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        logger.warning(f"OpenFIGI API request timed out for payload: {payload}")
    except requests.exceptions.HTTPError as http_err:
        logger.warning(f"OpenFIGI API HTTP error for payload {payload}: {http_err.response.status_code} - {http_err.response.text}")
    except requests.RequestException as req_err:
        logger.warning(f"OpenFIGI API request failed for payload {payload}: {req_err}")
    except json.JSONDecodeError as json_err:
        logger.error(f"Failed to decode JSON response from OpenFIGI for payload {payload}: {json_err}")
    return None