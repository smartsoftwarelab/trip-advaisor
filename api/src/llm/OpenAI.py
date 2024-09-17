import os

from langchain.schema import StrOutputParser

from langchain_community.chat_models import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

from wrapper.no_save_neo4j_chat_history_wrapper import NoSaveNeo4jChatMessageHistory
from wrapper.neo4j_client import Neo4jClient
from .basellm import BaseLLM


def get_session_history_method(save_conversation):
    if save_conversation:
        return get_save_session_history
    else:
        return get_no_save_session_history


def get_save_session_history(session_id):
    return Neo4jClient(
        base_url=os.getenv('NEO4J_CLIENT_URL'),
        session_id=session_id)


def get_no_save_session_history(session_id):
    return NoSaveNeo4jChatMessageHistory(
        base_url=os.getenv('NEO4J_CLIENT_URL'),
        session_id=session_id)


class ChatOpenAIChat(BaseLLM):
    def __init__(
            self,
            websocket,
            model_name: str = "gpt-3.5-turbo",
    ) -> None:
        self.websocket = websocket
        self.model = ChatOpenAI(openai_api_key=os.getenv('OPENAPI_APIKEY'),
                                openai_api_base=os.getenv('OPENAPI_BASEURL'),
                                streaming=True)
        self.model_name = model_name

    async def generate_streaming(
            self,
            question: str,
            session_id: str,
            similars,
            prompt: ChatPromptTemplate,
            send_response: bool = True,
            use_history: bool = True,
            save_conversation: bool = True
    ) -> [str]:

        await self.websocket.send_json({"type": "debug", "detail": f"created prompt: {prompt}"})
        await self.websocket.send_json({"type": "debug", "detail": f"fetched similars: {similars}"})

        chain = prompt | self.model | StrOutputParser()

        await self.websocket.send_json({"type": "debug", "detail": "chain created and model is going to generate"})

        session_history_method = get_session_history_method(save_conversation)

        tokens = []
        if use_history:
            chat_with_message_history = RunnableWithMessageHistory(
                chain,
                get_session_history=session_history_method,
                input_messages_key="question",
                history_messages_key="chat_history",
            )

            async for chunk in chat_with_message_history.astream(
                    {
                        "question": question,
                        "similars": similars,
                    },
                    config={
                        "configurable": {
                            "session_id": session_id
                        },
                    }
            ):
                response = {"type": "stream", "output": chunk}
                if send_response:
                    await self.websocket.send_json(response)

                tokens.append(chunk)
        else:
            async for chunk in chain.astream(
                    {
                        "question": question,
                        "similars": similars,
                    }
            ):
                response = {"type": "stream", "output": chunk}
                if send_response:
                    await self.websocket.send_json(response)

                tokens.append(chunk)

        final_response = self.reconstruct_streaming_response(tokens)

        return final_response
