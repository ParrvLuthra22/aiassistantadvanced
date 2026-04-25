"""ImageAgent - text-to-image generation using Stable Diffusion SDXL-Turbo."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from agents.base_agent import AgentCapability, BaseAgent
from schemas.events import HUDImageEvent, ImageGenerationEvent, VoiceOutputEvent

try:
    import torch
    from diffusers import AutoPipelineForText2Image

    SDXL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    AutoPipelineForText2Image = None  # type: ignore[assignment]
    SDXL_AVAILABLE = False


class ImageAgent(BaseAgent):
    """Generates images from prompts and publishes them to HUD + voice."""

    def __init__(
        self,
        name: Optional[str] = None,
        event_bus=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name or "ImageAgent", event_bus=event_bus, config=config)
        self.pipe = None
        self._load_lock = asyncio.Lock()
        self._loading_message_emitted = False

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="image_generation",
                description="Generate images from text prompts with SDXL-Turbo",
                input_events=["ImageGenerationEvent"],
                output_events=["HUDImageEvent", "VoiceOutputEvent"],
            )
        ]

    async def _setup(self) -> None:
        self._subscribe(ImageGenerationEvent, self._handle_generate)

    async def _teardown(self) -> None:
        self.pipe = None

    async def _handle_generate(self, event: ImageGenerationEvent) -> None:
        if not event.prompt.strip():
            await self._emit(
                VoiceOutputEvent(
                    text="I need a prompt before I can generate an image.",
                    source=self._name,
                    correlation_id=event.correlation_id or event.event_id,
                )
            )
            return

        try:
            await self._ensure_pipeline(event)
            if self.pipe is None:
                raise RuntimeError("Image model is unavailable")

            loop = asyncio.get_running_loop()
            prompt = event.prompt.strip()

            def _generate() -> str:
                image = self.pipe(
                    prompt=prompt,
                    num_inference_steps=1,
                    guidance_scale=0.0,
                ).images[0]
                path = f"/tmp/friday_gen_{int(time.time())}.png"
                image.save(path)
                return path

            path = await loop.run_in_executor(None, _generate)

            await self._emit(
                HUDImageEvent(
                    image_path=path,
                    source=self._name,
                    correlation_id=event.correlation_id or event.event_id,
                    metadata={"prompt": prompt},
                )
            )
            await self._emit(
                VoiceOutputEvent(
                    text=f"Here's your image of {prompt}",
                    source=self._name,
                    correlation_id=event.correlation_id or event.event_id,
                )
            )
        except Exception as exc:
            await self._emit(
                VoiceOutputEvent(
                    text=f"I couldn't generate that image: {exc}",
                    source=self._name,
                    correlation_id=event.correlation_id or event.event_id,
                )
            )

    async def _ensure_pipeline(self, event: ImageGenerationEvent) -> None:
        if self.pipe is not None:
            return

        if not SDXL_AVAILABLE or AutoPipelineForText2Image is None or torch is None:
            raise RuntimeError(
                "Image generation dependencies are missing. Install diffusers, transformers, accelerate, and torch."
            )

        async with self._load_lock:
            if self.pipe is not None:
                return

            if not self._loading_message_emitted:
                await self._emit(
                    VoiceOutputEvent(
                        text="Downloading image model, this will take a few minutes on first run",
                        source=self._name,
                        correlation_id=event.correlation_id or event.event_id,
                    )
                )
                self._loading_message_emitted = True

            loop = asyncio.get_running_loop()

            def _load():
                pipe = AutoPipelineForText2Image.from_pretrained(
                    "stabilityai/sdxl-turbo",
                    torch_dtype=torch.float16,
                    variant="fp16",
                )
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    pipe.to("mps")
                else:
                    pipe.to("cpu")
                return pipe

            self.pipe = await loop.run_in_executor(None, _load)
