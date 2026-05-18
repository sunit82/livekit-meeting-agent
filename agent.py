# agent.py
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    langchain,   # <-- this is key
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
    anam
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from graph import create_workflow  # <-- our compiled LangGraph app

load_dotenv(".env.local")

class MeetingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=(
            "You are a software project manager conducting a fortnightly meeting which has a set agenda "
            "The LangGraph workflow will drive the conversation flow. "
            "Simply speak the agenda items and responses as they come from the graph. "
            "Wait for the host to ask you to start the meeting and then proceed with the agenda items one by one as per the graph. "
            "Be conversational, professional, and helpful throughout the meeting process."
        ))

async def entrypoint(ctx: agents.JobContext):
    # 1) Build/compile the LangGraph app (Runnable)
    meeting_workflow = create_workflow()

    # 2) Wrap it as an LLM for LiveKit via the LangChain plugin
    #    (LLMAdapter knows how to drive LangGraph workflows as an LLM stream)
    lg_llm = langchain.LLMAdapter(graph=meeting_workflow)

    # 3) Configure the rest of the realtime pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=lg_llm,  # <-- use the adapter here instead of openai.LLM(...)
        tts=cartesia.TTS(model="sonic-2", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    # avatar = bey.AvatarSession(
    #     avatar_id="694c83e2-8895-4a98-bd16-56332ca3f449",  # ID of the Beyond Presence avatar to use
    # )

    avatar = anam.AvatarSession(
      persona_config=anam.PersonaConfig(
         name="Claudia",  # Name of the avatar to use.
         avatarId="bdaaedfa-00f2-417a-8239-8bb89adec682",  # ID of the avatar to use. See "Avatar setup" for details.
      ),
   )

    # Start the avatar and wait for it to join
    await avatar.start(session, room=ctx.room)

    await session.start(
        room=ctx.room,
        agent=MeetingAgent(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Start the Meeting workflow - the graph will drive the conversation
    print("Starting Meeting workflow...")
    # The graph will automatically begin with the first agenda item.

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))