from .neo4j_client import Neo4jClient


class NoSaveNeo4jChatMessageHistory(Neo4jClient):
    def add_message(self, message):
        # Override the save method to do nothing, preventing persistence
        pass
