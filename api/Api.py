import requests
from abc import ABC, abstractmethod
from enum import Enum

from requests import HTTPError, ConnectionError


class Method(Enum):
    GET = 1
    POST = 2

class Api(ABC):
    def __init__(self, host="localhost", port=8080, api_prefix=None, content_type="application/json"):
        self.host = host
        self.port = port
        self.api_prefix = api_prefix
        self.content_type = content_type
        self.headers = {'Content-Type': content_type}

    @property
    @abstractmethod
    def controller_prefix(self) -> str :
        return ""

    def __get_full_url(self, endpoint: str) -> str:
        base_url = f"http://{self.host}:{self.port}"
        if self.api_prefix is not None:
            base_url += f"/{self.api_prefix}"

        if self.controller_prefix is not "":
            base_url += f"/{self.controller_prefix}"

        return f"{base_url}/{endpoint}"

    def call(self, endpoint: str, method: Method, data: dict = None):
        try:
            if method == Method.GET:
                return self.__get(endpoint, data)
            elif method == Method.POST:
                return self.__post(endpoint, data)
        except ConnectionError as e:
            print(f"Connection Error: {str(e)}")
        except HTTPError as e:
            print(f"HTTP Error: {str(e)}")
        except TimeoutError as e:
            print(f"Request Timeout: {str(e)}")
        except ValueError as e:
            print(f"Invalid JSON Received: {str(e)}")


    def __get(self, endpoint: str, params: dict):
        response = requests.get(self.__get_full_url(endpoint), params=params, headers=self.headers)
        return response.json()

    def __post(self, endpoint: str, data: dict):
        response = requests.post(self.__get_full_url(endpoint), data=data, headers=self.headers)
        return response.json()
