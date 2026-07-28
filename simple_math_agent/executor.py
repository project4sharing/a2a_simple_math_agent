import os
from pathlib import Path
from dotenv import load_dotenv

import vertexai
from google.adk.runners import Runner

# A2A
from a2a.client import ClientConfig, ClientFactory
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a import types as a2a_types
from a2a.utils import TransportProtocol

# ADK
from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.sessions import VertexAiSessionService
from google.adk.tools import google_search_tool
from google.genai import types

# Agent Engine
from vertexai.agent_engines.templates.a2a import A2aAgent, create_agent_card

from .agent import root_agent as simple_math_agent

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path, verbose=True)

engine_id = os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID", "NULL_ENGINE_ID")
logger.info("#####  Using Google Cloud Agent Engine ID: %s", engine_id)

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT_ID", "NULL_PROJECT_ID")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "NULL_LOCATION")
MODEL_ARMOR_TEMPLATE = os.environ.get("MODEL_ARMOR_TEMPLATE", "NULL_MODEL_ARMOR_TEMPLATE")
BUCKET_URI = os.environ.get("DEPLOYMENT_GCS", "NULL_DEPLOYMENT_GCS")
FULL_MODEL_ARMOR_TEMPLATE_NAME = f"projects/{PROJECT_ID}/locations/{LOCATION}/templates/{MODEL_ARMOR_TEMPLATE}"
ENGINE_ID = os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID", "NULL_ENGINE_ID")

logger.info(f"#####  {env_path}")
logger.info("#####  Using Google Cloud Project ID: %s", PROJECT_ID)
logger.info("#####  Using Google Cloud Location: %s", LOCATION)
logger.info("#####  Using Model Armor Template: %s", MODEL_ARMOR_TEMPLATE)
logger.info("#####  Full Model Armor Template: %s", FULL_MODEL_ARMOR_TEMPLATE_NAME)

ENDPOINT = f"https://{LOCATION}-aiplatform.googleapis.com"

class SimpleMathAgentExecutor(AgentExecutor):
    """Refactored Executor using VertexAiSessionService for persistence."""

    def __init__(self) -> None:
        self.agent = None
        self.runner = None

    def _init_agent(self) -> None:
        if self.agent is None:
            # 1. Initialize Agent Platform using environment-injected metadata
            vertexai.init(project=PROJECT_ID, location=LOCATION)

            self.agent = simple_math_agent

            # 2. Initialize the Session Service
            # If engine_id exists, we are deployed remotely -> use VertexAiSessionService.
            # If engine_id is None, we are local -> use InMemorySessionService.
            if engine_id:
                session_service = VertexAiSessionService(
                    project=PROJECT_ID, location=LOCATION, agent_engine_id=ENGINE_ID
                )
            else:
                from google.adk.sessions.in_memory_session_service import (
                    InMemorySessionService,
                )

                session_service = InMemorySessionService()

            # 3. Setup Runner with the session service
            self.runner = Runner(
                app_name=self.agent.name,
                agent=self.agent,
                artifact_service=InMemoryArtifactService(),
                session_service=session_service,
                memory_service=InMemoryMemoryService(),
            )

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        if self.agent is None:
            self._init_agent()

        query = context.get_user_input()
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        user_id = (
            context.message.metadata["user_id"]
            if "user_id" in context.message.metadata
            else "vais-query-reasoning-engine"
        )

        task = a2a_types.Task(
            id=context.task_id,
            context_id=context.context_id,
            status=a2a_types.TaskStatus(
                state=a2a_types.TaskState.TASK_STATE_SUBMITTED
            ),
            history=[context.message] if context.message else [],
        )
        await event_queue.enqueue_event(task)

        await updater.start_work()

        try:
            # Using context_id (A2A) as session_id (Vertex) ensures continuity
            session = await self._get_or_create_session(context.context_id, user_id)

            content = types.Content(role="user", parts=[types.Part(text=query)])

            async for event in self.runner.run_async(
                session_id=session.id,
                user_id=user_id,
                new_message=content,
            ):
                if event.is_final_response():
                    answer = self._extract_answer(event)
                    await updater.add_artifact(
                        [a2a_types.Part(text=answer)],
                        name="answer",
                        last_chunk=True,
                    )
                    await updater.complete()
                    break

        except Exception as e:
            await updater.update_status(
                a2a_types.TaskState.TASK_STATE_FAILED,
                message=updater.new_agent_message(
                    [a2a_types.Part(text=f"An error occurred: {str(e)}")]
                ),
            )
            raise

    async def _get_or_create_session(self, context_id: str, user_id: str):
        engine_id = os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID")
        app_name = engine_id if engine_id else self.agent.name

        session = await self.runner.session_service.get_session(
            app_name=app_name,
            session_id=context_id,
            user_id=user_id,
        )

        if not session:
            session = await self.runner.session_service.create_session(
                app_name=app_name,
                user_id=user_id,
            )
        return session

    def _extract_answer(self, event) -> str:
        parts = event.content.parts
        text_parts = [part.text for part in parts if part.text]
        return " ".join(text_parts) if text_parts else "No answer found."

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle task cancellation requests."""
        task_id = context.task_id
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task_id or "",
            context_id=context.context_id or "",
        )
        await updater.cancel()