import os
import requests
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, convert_to_messages

# API Endpoints
chat_history_url = os.getenv('NEO4J_CLIENT_URL_CHATHISTORY')
query_url = os.getenv('NEO4J_CLIENT_URL_QUERY')
get_city_url = os.getenv('NEO4J_CLIENT_URL_GET_CITY')
find_nearest_cities_url = os.getenv('NEO4J_CLIENT_URL_FIND_NEAREST_CITIES')
get_attractions_url = os.getenv('NEO4J_CLIENT_URL_GET_ATTRACTION')


class Neo4jClient(BaseChatMessageHistory):
    def __init__(self, base_url: str, session_id: str) -> None:
        if not base_url:
            raise Exception("Base URL of Neo4j client is not provided.")
        self.base_url = base_url
        self.session_id = session_id
        self.headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        self.session = requests.Session()  # Persistent session for connection reuse

    def clear(self) -> None:
        """Clear chat history for a session."""
        body_request = {'session_id': self.session_id}
        response = self.session.delete(f"{self.base_url}{chat_history_url}", json=body_request, headers=self.headers)
        return response.json()

    @property
    def messages(self) -> List[BaseMessage]:
        """Get chat history messages for a session."""
        response = self.session.get(f"{self.base_url}{chat_history_url}/{self.session_id}", headers=self.headers)
        result = response.json().get("messages")
        return convert_to_messages(result)

    @messages.setter
    def messages(self, messages: List[BaseMessage]) -> None:
        raise NotImplementedError("Direct assignment to 'messages' is not allowed. Use 'add_message' instead.")

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the chat history."""
        message_summery = [
            {
                "type": message.type,
                "content": message.content
            }
        ]

        body_request = {'session_id': self.session_id, 'message': message_summery}
        response = self.session.post(f"{self.base_url}{chat_history_url}", json=body_request, headers=self.headers)

    def query(self, query: str):
        """Perform a Neo4j query."""
        response = self.session.get(f"{self.base_url}{query_url}", params={'q': query}, headers=self.headers)
        return response.json().get('result')

    def get_city(self, city_name: str):
        """Get information about a city."""
        response = self.session.get(f"{self.base_url}{get_city_url}/{city_name}", headers=self.headers)
        return response.json().get('city')

    def find_nearest_cities(self, city_name: str):
        """Find nearest cities."""
        response = self.session.get(f"{self.base_url}{find_nearest_cities_url}/{city_name}", headers=self.headers)
        return response.json().get('nearest_cities')

    def get_attractions(self, city_names: List[str]):
        """Get attractions for a list of cities."""
        body_request = {'city_names': city_names}
        response = self.session.post(f"{self.base_url}{get_attractions_url}", json=body_request, headers=self.headers)
        return response.json().get('attractions')

    def __del__(self) -> None:
        """Close the session on object destruction."""
        self.session.close()
