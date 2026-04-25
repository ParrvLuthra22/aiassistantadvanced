"""VisionAgent - screenshot-driven vision workflows via Gemini."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import ImageGrab

try:
    import pyautogui

    PYAUTOGUI_AVAILABLE = True
except Exception:  # pragma: no cover - environment-dependent import
    pyautogui = None
    PYAUTOGUI_AVAILABLE = False

try:
    import google.generativeai as genai

    GEMINI_SDK_AVAILABLE = True
except Exception:  # pragma: no cover - dependency/environment dependent
    genai = None
    GEMINI_SDK_AVAILABLE = False

from agents.base_agent import AgentCapability, BaseAgent
from schemas.events import (
    HUDUpdateEvent,
    ImageGenerationEvent,
    IntentRecognizedEvent,
    ScreenshotEvent,
    VoiceOutputEvent,
)


SCREENSHOT_PATH = Path("/tmp/jarvis_screen.png")
GEMINI_MODEL_NAME = "gemini-1.5-flash"
MIN_CLICK_CONFIDENCE = 0.7


@dataclass
class CoordinateResult:
    x: Optional[int]
    y: Optional[int]
    confidence: float
    raw: str = ""

    @property
    def has_coordinates(self) -> bool:
        return self.x is not None and self.y is not None


class VisionAgent(BaseAgent):
    """Agent that captures screenshots and delegates visual reasoning to Gemini."""

    def __init__(
        self,
        name: Optional[str] = None,
        event_bus=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name or "VisionAgent", event_bus=event_bus, config=config)
        self._model = None
        self._gemini_enabled = False

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="screen_understanding",
                description="Captures screen, answers visual questions, and performs safe click actions",
                input_events=["IntentRecognizedEvent", "ScreenshotEvent"],
                output_events=["VoiceOutputEvent", "HUDUpdateEvent", "ImageGenerationEvent"],
            )
        ]

    async def _setup(self) -> None:
        self._configure_pyautogui()
        self._initialize_gemini()
        self._subscribe(IntentRecognizedEvent, self._handle_intent)
        self._subscribe(ScreenshotEvent, self._handle_screenshot_event)

    async def _teardown(self) -> None:
        self._model = None
        self._gemini_enabled = False

    def _configure_pyautogui(self) -> None:
        if PYAUTOGUI_AVAILABLE and pyautogui is not None:
            pyautogui.FAILSAFE = True

    def _initialize_gemini(self) -> None:
        if not GEMINI_SDK_AVAILABLE:
            self._logger.warning(
                "google-generativeai SDK unavailable. Install it to enable visual reasoning."
            )
            return

        api_key = (
            self._get_config("vision.gemini.api_key")
            or self._get_config("intent.gemini.api_key")
            or os.getenv("GEMINI_API_KEY")
        )
        if not api_key:
            self._logger.warning("Gemini API key not configured for VisionAgent")
            return

        try:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
            self._gemini_enabled = True
            self._logger.info(f"Vision Gemini initialized with model={GEMINI_MODEL_NAME}")
        except Exception as exc:  # pragma: no cover - remote SDK init
            self._logger.error(f"Failed to initialize Gemini Vision client: {exc}")

    async def _handle_screenshot_event(self, event: ScreenshotEvent) -> None:
        try:
            if event.bbox:
                await self.capture_region(*event.bbox)
            else:
                await self.capture_full_screen()
            await self._emit(
                HUDUpdateEvent(
                    image_path=str(SCREENSHOT_PATH),
                    source=self._name,
                    correlation_id=event.correlation_id or event.event_id,
                )
            )
        except Exception as exc:
            await self._emit(
                VoiceOutputEvent(
                    text=f"I couldn't capture the screen: {exc}",
                    source=self._name,
                    correlation_id=event.correlation_id or event.event_id,
                )
            )

    async def _handle_intent(self, event: IntentRecognizedEvent) -> None:
        intent_text = (event.intent or "").strip().lower()
        raw_text = (event.raw_text or "").strip().lower()
        candidates = [intent_text, raw_text]

        if self._matches_any(candidates, ["what's on my screen", "read my screen", "describe screen"]):
            await self._answer_visual_prompt(
                event=event,
                prompt="Describe everything visible on this screen in detail",
            )
            return

        if self._matches_any(candidates, ["read that", "what does it say"]):
            await self._answer_visual_prompt(
                event=event,
                prompt="Read all text visible on this screen",
            )
            return

        find_query = self._extract_query(candidates, [r"find\s+(.+?)\s+on\s+screen", r"where is\s+(.+)"])
        if find_query:
            await self._answer_visual_prompt(
                event=event,
                prompt=f"Where is {find_query} on this screen? Give pixel coordinates if possible",
            )
            return

        click_query = self._extract_query(candidates, [r"click\s+(.+)"])
        if click_query:
            await self._click_query_target(click_query, event)
            return

        image_query = self._extract_query(
            candidates,
            [r"visuali[sz]e\s+(.+)", r"show me\s+(.+)", r"generate image of\s+(.+)"],
        )
        if image_query:
            await self._emit(
                ImageGenerationEvent(
                    prompt=image_query,
                    source=self._name,
                    correlation_id=event.correlation_id or event.event_id,
                )
            )

    @staticmethod
    def _matches_any(candidates: List[str], phrases: List[str]) -> bool:
        return any(any(phrase in candidate for phrase in phrases) for candidate in candidates if candidate)

    @staticmethod
    def _extract_query(candidates: List[str], patterns: List[str]) -> Optional[str]:
        for candidate in candidates:
            if not candidate:
                continue
            for pattern in patterns:
                match = re.search(pattern, candidate)
                if match:
                    query = match.group(1).strip().strip("?.!,")
                    if query:
                        return query
        return None

    async def _answer_visual_prompt(self, event: IntentRecognizedEvent, prompt: str) -> None:
        try:
            await self.capture_full_screen()
            await self._emit(
                HUDUpdateEvent(
                    image_path=str(SCREENSHOT_PATH),
                    source=self._name,
                    correlation_id=event.correlation_id or event.event_id,
                )
            )
            response_text = await self._ask_gemini(prompt)
        except Exception as exc:
            response_text = f"I couldn't analyze the screen: {exc}"

        await self._emit(
            VoiceOutputEvent(
                text=response_text,
                source=self._name,
                correlation_id=event.correlation_id or event.event_id,
            )
        )

    async def _click_query_target(self, query: str, event: IntentRecognizedEvent) -> None:
        if not PYAUTOGUI_AVAILABLE or pyautogui is None:
            await self._emit(
                VoiceOutputEvent(
                    text="Click actions are unavailable because pyautogui is not installed.",
                    source=self._name,
                    correlation_id=event.correlation_id or event.event_id,
                )
            )
            return

        try:
            await self.capture_full_screen()
            await self._emit(
                HUDUpdateEvent(
                    image_path=str(SCREENSHOT_PATH),
                    source=self._name,
                    correlation_id=event.correlation_id or event.event_id,
                )
            )
            prompt = (
                f"Find '{query}' on this screen and provide pixel coordinates. "
                "Return strict JSON with keys x, y, confidence. "
                "Confidence must be a number between 0 and 1."
            )
            response_text = await self._ask_gemini(prompt)
            parsed = self._parse_coordinates(response_text)

            if not parsed.has_coordinates:
                message = (
                    "I could not find clickable coordinates for that target. "
                    f"Model response: {response_text}"
                )
            elif parsed.confidence < MIN_CLICK_CONFIDENCE:
                message = (
                    f"I found possible coordinates ({parsed.x}, {parsed.y}) but confidence "
                    f"{parsed.confidence:.2f} is below 0.70, so I did not click."
                )
            else:
                clicked = await self._safe_click(parsed.x, parsed.y)
                if clicked:
                    message = (
                        f"Clicked {query} at ({parsed.x}, {parsed.y}) with "
                        f"confidence {parsed.confidence:.2f}."
                    )
                else:
                    message = "I found the target but the click action failed safely."
        except Exception as exc:
            message = f"I couldn't complete that click action: {exc}"

        await self._emit(
            VoiceOutputEvent(
                text=message,
                source=self._name,
                correlation_id=event.correlation_id or event.event_id,
            )
        )

    async def capture_full_screen(self) -> str:
        """Capture full screen using PIL.ImageGrab.grab()."""
        return await self._capture_screen()

    async def capture_region(self, x: int, y: int, w: int, h: int) -> str:
        """Capture a region using grab(bbox=(x, y, w, h))."""
        return await self._capture_screen((x, y, w, h))

    async def _capture_screen(
        self,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> str:
        def _do_capture() -> str:
            if bbox is None:
                image = ImageGrab.grab()
            else:
                x, y, w, h = bbox
                # Supports both (x, y, right, bottom) and (x, y, width, height).
                pil_bbox = (x, y, w, h)
                if w <= x or h <= y:
                    pil_bbox = (x, y, x + w, y + h)
                image = ImageGrab.grab(bbox=pil_bbox)
            image.save(SCREENSHOT_PATH)
            return str(SCREENSHOT_PATH)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _do_capture)

    async def _ask_gemini(self, prompt: str) -> str:
        if not self._gemini_enabled or self._model is None:
            raise RuntimeError("Gemini Vision is not configured")

        def _run_request() -> str:
            image_bytes = SCREENSHOT_PATH.read_bytes()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            contents = [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": image_b64,
                    }
                },
            ]
            response = self._model.generate_content(contents)
            text = getattr(response, "text", "")
            return (text or "").strip() or "I analyzed the screen but received no text response."

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _run_request)

    async def _safe_click(self, x: Optional[int], y: Optional[int]) -> bool:
        if x is None or y is None or not PYAUTOGUI_AVAILABLE or pyautogui is None:
            return False

        def _do_click() -> bool:
            try:
                pyautogui.click(x=x, y=y)
                return True
            except Exception as exc:
                self._logger.warning(f"Safe click failed: {exc}")
                return False

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _do_click)

    @staticmethod
    def _parse_coordinates(response_text: str) -> CoordinateResult:
        text = (response_text or "").strip()
        if not text:
            return CoordinateResult(x=None, y=None, confidence=0.0, raw=response_text)

        # Try direct JSON first.
        json_candidate = text
        if "{" in text and "}" in text:
            json_candidate = text[text.find("{") : text.rfind("}") + 1]

        try:
            payload = json.loads(json_candidate)
            x = int(payload.get("x")) if payload.get("x") is not None else None
            y = int(payload.get("y")) if payload.get("y") is not None else None
            confidence = float(payload.get("confidence", 0.0))
            return CoordinateResult(x=x, y=y, confidence=confidence, raw=text)
        except Exception:
            pass

        coord_match = re.search(r"\(?\s*(\d{1,5})\s*,\s*(\d{1,5})\s*\)?", text)
        x = int(coord_match.group(1)) if coord_match else None
        y = int(coord_match.group(2)) if coord_match else None

        conf_match = re.search(r"confidence[^0-9]*(1(?:\.0+)?|0?\.\d+|\d{1,3}%?)", text, flags=re.IGNORECASE)
        confidence = 0.0
        if conf_match:
            token = conf_match.group(1)
            if token.endswith("%"):
                confidence = float(token.rstrip("%")) / 100.0
            else:
                confidence = float(token)
                if confidence > 1.0:
                    confidence = confidence / 100.0

        return CoordinateResult(x=x, y=y, confidence=confidence, raw=text)
