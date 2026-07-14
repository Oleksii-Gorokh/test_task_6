from livekit.agents import AgentServer, JobContext, cli

from confidence_voice_agent.agent import ConfidenceAwareAgent
from confidence_voice_agent.config import Settings, get_settings
from confidence_voice_agent.logging import configure_logging
from confidence_voice_agent.providers import create_agent_session


def create_server(settings: Settings) -> AgentServer:
    server = AgentServer(
        ws_url=settings.livekit_url,
        api_key=settings.livekit_api_key.get_secret_value(),
        api_secret=settings.livekit_api_secret.get_secret_value(),
        log_level=settings.log_level,
    )

    @server.rtc_session()
    async def entrypoint(ctx: JobContext) -> None:
        session = create_agent_session(settings)
        await session.start(
            agent=ConfidenceAwareAgent(settings),
            room=ctx.room,
        )

    return server


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    cli.run_app(create_server(settings))


if __name__ == "__main__":
    main()
