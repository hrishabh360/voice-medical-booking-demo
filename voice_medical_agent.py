# agent.py
from __future__ import annotations
from typing import Any, Optional
from datetime import datetime, timedelta
import uuid

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, RunContext, function_tool
# from livekit.plugins import noise_cancellation, silero
# from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Load LIVEKIT_* and OPENAI_API_KEY from .env.local
load_dotenv(".env.local")

# ---------------------------
# Simple in-memory “backend”
# ---------------------------
ACCEPTED_INSURERS = {"Aetna", "Blue Cross", "Cigna", "Medicare"}
BOOKINGS: list[dict[str, Any]] = []

def _default_slots(n_days: int = 3) -> list[dict]:
    """Generate 2–3 slots/day for the next few days (local demo)."""
    now = datetime.now()
    slots = []
    daily = [(10, 0), (14, 30), (16, 0)]
    for d in range(n_days):
        base = (now + timedelta(days=d+1)).replace(hour=0, minute=0, second=0, microsecond=0)
        for hh, mm in daily:
            start = base.replace(hour=hh, minute=mm)
            slots.append({"start_iso": start.isoformat(), "duration_min": 20})
    return slots

SLOTS = _default_slots()

SYSTEM_PROMPT = """
You are a friendly medical office scheduling agent.

Your job:
1) Ask the caller, in natural language, what symptoms or health issue they have. Ask 1–2 brief follow-up questions (onset, severity, key symptom) to capture a concise reason for visit.
2) Ask which insurance they have. Use tool:check_insurance to verify acceptance. If not accepted, offer self-pay or suggest they call the front desk; ask if they still want to continue.
3) Offer 2–3 appointment options using tool:get_slots (respect date preference if provided).
4) Ask for full name and phone. Then call tool:create_booking.
5) Read back a short summary and the confirmation ID.

Style:
- One question at a time. Be concise and courteous.
- Confirm critical details (insurer name, chosen time, phone).
- This is a demo: do not collect sensitive medical history. Keep “reason for visit” short.
"""

# ---------------------------
# The Agent with tools
# ---------------------------
class BookingAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)

    @function_tool(description="Verify whether the clinic accepts an insurance by name.")
    async def check_insurance(self, context: RunContext, insurer_name: str) -> dict[str, Any]:
        """
        Args:
            insurer_name: The insurer the caller names (e.g., "Blue Cross").
        Returns:
            {"accepted": bool}
        """
        accepted = insurer_name.strip() in ACCEPTED_INSURERS
        return {"accepted": accepted}

    @function_tool(description="Return a short list of available appointment slots.")
    async def get_slots(self, context: RunContext, date_pref: Optional[str] = None) -> dict[str, Any]:
        """
        Args:
            date_pref: Optional YYYY-MM-DD preferred date.
        Returns:
            {"slots": [{"start_iso": "...", "duration_min": 20}, ...]}
        """
        if date_pref:
            pref = [s for s in SLOTS if s["start_iso"].startswith(date_pref)]
            if pref:
                return {"slots": pref[:3]}
        return {"slots": SLOTS[:6]}  # a few options

    @function_tool(description="Create an appointment booking once details are confirmed.")
    async def create_booking(
        self,
        context: RunContext,
        name: str,
        phone: str,
        insurer: str,
        reason: str,
        slot_iso: str,
    ) -> dict[str, Any]:
        """
        Args:
            name: Full name of patient
            phone: Phone digits
            insurer: Insurance name
            reason: Short reason for visit
            slot_iso: ISO-8601 start time (must be from get_slots)
        Returns:
            {"confirmation_id": "ABC123", "saved": true}
        """
        # basic validation
        if slot_iso not in {s["start_iso"] for s in SLOTS}:
            return {"error": "slot_unavailable"}

        confirmation = uuid.uuid4().hex[:8].upper()
        BOOKINGS.append(
            {"id": confirmation, "name": name, "phone": phone, "insurer": insurer, "reason": reason, "slot_iso": slot_iso}
        )
        return {"confirmation_id": confirmation, "saved": True}

# ---------------------------
# LiveKit session entrypoint
# ---------------------------
async def entrypoint(ctx: agents.JobContext):
    """
    Start a voice pipeline with STT+LLM+TTS, VAD + turn detection, and noise cancellation,
    per LiveKit’s quickstart pattern.
    """
    session = AgentSession(
        # Models: use defaults from docs; you can change to your preferred providers
        stt="assemblyai/universal-streaming:en",    # Speech-to-text
        llm="groq/llama-3.1-70b-versatile",                   # or another provider via plugin
        tts="cartesia/sonic-2:default",            # Text-to-speech
        vad=None,
        turn_detection=None,
    )

    await session.start(
        room=ctx.room,
        agent=BookingAgent(),
        room_input_options=RoomInputOptions(
            # noise_cancellation=noise_cancellation.BVC()  # great defaults for mic/phone audio
        ),
    )
# --- CLI entry ---
if __name__ == "__main__":
    from livekit.agents import cli
    from livekit.agents.worker import WorkerOptions

    # pass options, not the function itself
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))



