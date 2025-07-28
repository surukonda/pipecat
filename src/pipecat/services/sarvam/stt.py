#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sarvam AI speech-to-text service implementation."""

import io
import wave
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

# Language mapping for Sarvam AI
SARVAM_LANGUAGE_MAP = {
    Language.BN_IN: "bn-IN",  # Bengali
    Language.EN_IN: "en-IN",  # English (India)
    Language.GU_IN: "gu-IN",  # Gujarati
    Language.HI_IN: "hi-IN",  # Hindi
    Language.KN_IN: "kn-IN",  # Kannada
    Language.ML_IN: "ml-IN",  # Malayalam
    Language.MR_IN: "mr-IN",  # Marathi
    Language.OR_IN: "or-IN",  # Odia
    Language.PA_IN: "pa-IN",  # Punjabi
    Language.TA_IN: "ta-IN",  # Tamil
    Language.TE_IN: "te-IN",  # Telugu
}


def language_to_sarvam_language(language: Language) -> Optional[str]:
    """Convert Pipecat Language enum to Sarvam AI language codes.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Sarvam AI language code, or None if not supported.
    """
    return SARVAM_LANGUAGE_MAP.get(language)


def sarvam_language_to_language(sarvam_language: str) -> Optional[Language]:
    """Convert Sarvam AI language code back to Pipecat Language enum.

    Args:
        sarvam_language: The Sarvam AI language code to convert.

    Returns:
        The corresponding Language enum, or None if not supported.
    """
    for lang_enum, lang_code in SARVAM_LANGUAGE_MAP.items():
        if lang_code == sarvam_language:
            return lang_enum
    return None


class SarvamSTTService(SegmentedSTTService):
    """Speech-to-Text service using Sarvam AI's API.

    Converts speech to text using Sarvam AI's STT models with support for multiple
    Indian languages. This is a segmented STT service that processes speech in
    segments using VAD events.

    Example::

        stt = SarvamSTTService(
            api_key="your-api-key",
            model="saarika:v2.5",
            aiohttp_session=session,
            params=SarvamSTTService.InputParams(
                language=Language.HI_IN,
                with_timestamps=True
            )
        )
    """

    class InputParams(BaseModel):
        """Input parameters for Sarvam STT configuration.

        Parameters:
            language: Language for transcription. Defaults to English (India).
            with_timestamps: Whether to include timestamps in the response. Defaults to False.
            model: STT model to use. Defaults to "saarika:v2.5".
        """

        language: Optional[Language] = Language.EN_IN
        with_timestamps: Optional[bool] = False
        model: Optional[str] = "saarika:v2.5"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "saarika:v2.5",
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = "https://api.sarvam.ai",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Sarvam STT service.

        Args:
            api_key: Sarvam AI API subscription key.
            model: STT model to use. Defaults to "saarika:v2.5".
            aiohttp_session: Shared aiohttp session for making requests.
            base_url: Sarvam AI API base URL. Defaults to "https://api.sarvam.ai".
            sample_rate: Audio sample rate in Hz. If None, uses default.
            params: Additional STT parameters. If None, uses defaults.
            **kwargs: Additional arguments passed to parent SegmentedSTTService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or SarvamSTTService.InputParams()

        self._api_key = api_key
        self._base_url = base_url
        self._session = aiohttp_session

        self._settings = {
            "language": self.language_to_service_language(params.language)
            if params.language
            else "en-IN",
            "with_timestamps": params.with_timestamps,
            "model": params.model or model,
        }

        self.set_model_name(self._settings["model"])

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Sarvam service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Sarvam AI language format.

        Args:
            language: The language to convert.

        Returns:
            The Sarvam AI-specific language code, or None if not supported.
        """
        return language_to_sarvam_language(language)

    async def set_language(self, language: Language):
        """Set the language for speech recognition.

        Args:
            language: The language to use for speech recognition.

        Raises:
            ValueError: If the language is not supported by Sarvam AI.
        """
        logger.info(f"Switching STT language to: [{language}]")
        sarvam_language = self.language_to_service_language(language)
        if sarvam_language:
            self._settings["language"] = sarvam_language
        else:
            supported_languages = list(SARVAM_LANGUAGE_MAP.keys())
            raise ValueError(
                f"Language {language} not supported by Sarvam AI. Supported languages: {supported_languages}"
            )

    async def start(self, frame: StartFrame):
        """Start the Sarvam STT service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        # This method is called to enable tracing of transcription events
        # The actual transcription frame is yielded in run_stt method
        logger.debug(
            f"Handled transcription: {transcript} (final: {is_final}, language: {language})"
        )

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Generate transcription from audio using Sarvam AI's API.

        Args:
            audio: Raw audio bytes to transcribe (in WAV format).

        Yields:
            Frame: Transcription frames containing the recognized text.
        """
        logger.debug(f"{self}: Generating STT transcription")

        try:
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()

            # Create form data using aiohttp.FormData (not requests-style files dict)
            form_data = aiohttp.FormData()
            form_data.add_field("file", audio, filename="audio.wav", content_type="audio/wav")

            # Add model and language_code to the form data
            form_data.add_field("model", self._settings["model"])
            form_data.add_field("language_code", self._settings["language"])

            # Add optional parameters
            if self._settings["with_timestamps"]:
                form_data.add_field("with_timestamps", "true")

            headers = {
                "api-subscription-key": self._api_key,
            }

            # Construct the full URL with the speech-to-text endpoint
            url = f"{self._base_url}/speech-to-text"

            async with self._session.post(url, data=form_data, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Sarvam STT API error: {error_text}")
                    yield ErrorFrame(f"Sarvam STT API error: {error_text}")
                    return

                response_data = await response.json()

            await self.stop_ttfb_metrics()

            # Extract transcript from response
            if "transcript" not in response_data:
                logger.error("No transcript data received from Sarvam API")
                yield ErrorFrame("No transcript data received")
                return

            transcript = response_data["transcript"].strip()

            if transcript:
                # Convert language code back to Language enum for _handle_transcription
                language_enum = sarvam_language_to_language(self._settings["language"])
                await self._handle_transcription(transcript, True, language_enum)
                logger.debug(f"Transcription: [{transcript}]")
                yield TranscriptionFrame(
                    transcript,
                    self._user_id,
                    time_now_iso8601(),
                    result=response_data,
                )
            else:
                logger.warning("Received empty transcription from Sarvam API")

        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(f"Error generating STT: {e}")
        finally:
            await self.stop_processing_metrics()
