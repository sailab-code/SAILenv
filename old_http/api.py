#
# Copyright (C) 2020 Enrico Meloni, Luca Pasqualini, Matteo Tiezzi
# University of Siena - Artificial Intelligence Laboratory - SAILab
#
#
# SAILenv is licensed under a MIT license.
#
# You should have received a copy of the license along with this
# work. If not, see <https://en.wikipedia.org/wiki/MIT_License>.

# Import packages

from requests import HTTPError, ConnectionError, post, get
from enum import Enum


class Method(Enum):
    """
    Enum defining GET (1) and POST (2) HTTP methods.
    """
    GET = 1
    POST = 2


class Api:
    """
    Base abstract class for all APIs.
    """
    def __init__(self,
                 host: str = "localhost", port: int = 8080,
                 api_prefix: str = None, content_type: str = "application/json"):
        self.host: str = host
        self.port: int = port
        self.api_prefix: str = api_prefix
        self.content_type: str = content_type
        self.headers: {} = {'Content-Type': content_type}

    def __get(self, endpoint: str, params: dict):
        """
        Call the GET HTTP method with the given end-point string and parameters.

        :param endpoint: the last part of base_url/endpoint
        :param params: a dictionary of parameters
        :return: the json of the response
        """
        response = get(self.__get_full_url(endpoint), params=params, headers=self.headers)
        return response.json(), response.elapsed.total_seconds()

    def __post(self, endpoint: str, data: dict):
        """
        Call the POST HTTP method with the given end-point string and data.

        :param endpoint: the last part of base_url/endpoint
        :param data: a dictionary of data
        :return: the json of the response
        """
        response = post(self.__get_full_url(endpoint), json=data, headers=self.headers)
        return response.json()

    def __get_full_url(self, endpoint: str) -> str:
        """
        Get the full URL of the HTTP API with the given end-point string (format base_url/endpoint)

        :return: the full URL as string
        """
        base_url = f"old_http://{self.host}:{self.port}"

        if self.api_prefix is not None:
            base_url += f"/{self.api_prefix}"

        if self.controller_prefix is not "":
            base_url += f"/{self.controller_prefix}"

        return f"{base_url}/{endpoint}"

    def call(self, endpoint: str,
             method: Method,
             data: dict = None):
        """
        Call the API on its URL with the given end-point string and method, passing the given dict of data as parameters.

        :param endpoint: the last part of base_url/endpoint
        :param method: GET or POST HTTP method
        :param data: a dict of parameters
        :return: the result of the call, raise an exception for: ConnectionError, HTTPError, TimeoutError or ValueError
        """
        try:
            if method == Method.GET:
                return self.__get(endpoint, data)
            elif method == Method.POST:
                return self.__post(endpoint, data)
        except ConnectionError as e:
            raise e
        except HTTPError as e:
            raise ConnectionError(e)
        except TimeoutError as e:
            raise ConnectionError(e)
        except ValueError as e:
            raise ConnectionError(e)

    @property
    def controller_prefix(self):
        # Abstract property, should be implemented by children
        return NotImplementedError()
