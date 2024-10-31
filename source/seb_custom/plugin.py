#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
unmanic-plugins.plugin.py

Written by: ds-sebastian
Date: October 30, 2024

This plugin provides comprehensive audio stream management for media files,
ensuring optimal compatibility with modern devices and streaming services
while maintaining quality.

Features:
- Automatically manages sample rates (converts to 48kHz when needed)
- Intelligently handles multichannel audio (5.1, 7.1)
- Creates stereo downmixes of multichannel content when appropriate
- Applies volume normalization separately for stereo and surround content
- Preserves all language tracks and metadata
- Optimizes bitrates based on channel count
"""

import logging
import os
from typing import Dict, List

from seb_custom.lib.ffmpeg import Parser, Probe, StreamMapper
from unmanic.libs.directoryinfo import UnmanicDirectoryInfo
from unmanic.libs.unplugins.settings import PluginSettings

# Configure plugin logger
logger = logging.getLogger("Unmanic.Plugin.seb_custom")


class Settings(PluginSettings):
    """Plugin settings and UI configuration"""

    settings = {
        # General processing options
        "keep_original_streams": True,  # Always keep originals for compatibility and quality
        "force_processing": False,  # No need to force by default
        # Sample rate handling - your TV's limit
        "target_sample_rate": 48000,  # 48kHz max for TV compatibility
        # Audio encoding - matches flow requirements
        "stereo_encoder": "libfdk_aac",  # Better quality for stereo
        "multichannel_71_encoder": "opus",  # Required for >6 channels
        "multichannel_51_encoder": "libfdk_aac",  # Better quality for 5.1
        # Stereo downmix - always create for compatibility
        "create_stereo_tracks": True,
        "stereo_track_suffix": "AAC Stereo",
        # Loudness normalization - exact values from flow
        "enable_loudnorm": True,
        "surround_norm_i": "-24.0",  # Surround target LUFS
        "surround_norm_lra": "7.0",  # Surround loudness range
        "surround_norm_tp": "-2.0",  # Surround true peak
        "stereo_norm_i": "-16.0",  # Stereo target LUFS (streaming standard)
        "stereo_norm_lra": "11.0",  # Stereo loudness range
        "stereo_norm_tp": "-1.0",  # Stereo true peak
        # Bitrate calculations handled in code (64k per channel)
        # Stream ordering handled in code (main first, stereo second)
        # Advanced
        "max_muxing_queue_size": 2048,  # Safe default for complex audio
    }

    def __init__(self, *args, **kwargs):
        super(Settings, self).__init__(*args, **kwargs)
        self.form_settings = {
            "keep_original_streams": {
                "label": "Keep original audio streams",
                "description": "Always keep original high-quality streams alongside stereo versions",
            },
            "force_processing": {
                "label": "Force reprocessing",
                "description": "Process streams even if they already meet requirements",
            },
            "target_sample_rate": {
                "label": "Maximum sample rate",
                "description": "Convert higher sample rates to this value (48kHz recommended for device compatibility)",
                "input_type": "select",
                "select_options": [
                    {"value": 48000, "label": "48 kHz (Recommended)"},
                    {"value": 96000, "label": "96 kHz"},
                    {"value": 192000, "label": "192 kHz"},
                ],
            },
            "stereo_encoder": {
                "label": "Stereo/5.1 encoder",
                "description": "Encoder for stereo and 5.1 content (libfdk_aac recommended for quality)",
                "input_type": "select",
                "select_options": [
                    {"value": "libfdk_aac", "label": "libfdk_aac (Better Quality)"},
                    {"value": "opus", "label": "Opus (Better Compatibility)"},
                    {"value": "aac", "label": "Native AAC (Fallback)"},
                ],
            },
            "multichannel_71_encoder": {
                "label": "7.1 Channel encoder",
                "description": "Encoder for 7.1 content (Opus recommended to preserve all channels)",
                "input_type": "select",
                "select_options": [
                    {"value": "opus", "label": "Opus (Recommended for >6 channels)"},
                    {
                        "value": "libfdk_aac",
                        "label": "libfdk_aac (Will downmix to 5.1)",
                    },
                ],
            },
            "create_stereo_tracks": {
                "label": "Create stereo versions",
                "description": "Create stereo AAC versions of all multichannel streams for maximum compatibility",
            },
            "enable_loudnorm": {
                "label": "Enable loudness normalization",
                "description": "Apply professional loudness standards (-24 LUFS for surround, -16 LUFS for stereo)",
            },
            "surround_norm_i": {
                "label": "Surround loudness target (LUFS)",
                "description": "Target integrated loudness for surround content (-24 LUFS recommended)",
                "input_type": "slider",
                "slider_options": {
                    "min": -70.0,
                    "max": -5.0,
                    "step": 0.1,
                },
            },
            "stereo_norm_i": {
                "label": "Stereo loudness target (LUFS)",
                "description": "Target integrated loudness for stereo content (-16 LUFS recommended for streaming)",
                "input_type": "slider",
                "slider_options": {
                    "min": -70.0,
                    "max": -5.0,
                    "step": 0.1,
                },
            },
            # Hide more technical settings by default
            "surround_norm_lra": {
                "label": "Surround loudness range",
                "display": "advanced",
            },
            "surround_norm_tp": {"label": "Surround true peak", "display": "advanced"},
            "stereo_norm_lra": {
                "label": "Stereo loudness range",
                "display": "advanced",
            },
            "stereo_norm_tp": {"label": "Stereo true peak", "display": "advanced"},
            "max_muxing_queue_size": {
                "label": "Muxing queue size",
                "display": "advanced",
            },
        }


class AudioStreamInfo:
    """Helper class to store analyzed stream information"""

    def __init__(self, stream_info: dict):
        self.channels = int(stream_info.get("channels", 2))
        self.sample_rate = int(stream_info.get("sample_rate", 48000))
        self.language = stream_info.get("tags", {}).get("language", "und")
        self.codec_name = stream_info.get("codec_name", "").lower()
        self.bit_rate = int(stream_info.get("bit_rate", 128000))
        self.is_atmos = self._detect_atmos(stream_info)
        self.stream_index = stream_info.get("index", 0)
        self.title = stream_info.get("tags", {}).get("title", "")

    def _detect_atmos(self, stream_info: dict) -> bool:
        """Detect if stream is Atmos/TrueHD"""
        codec_name = stream_info.get("codec_name", "").lower()
        format_name = stream_info.get("format_name", "").lower()
        return "truehd" in codec_name or "atmos" in format_name


class PluginStreamMapper(StreamMapper):
    """Handles stream analysis and processing decisions"""

    def __init__(self):
        super(PluginStreamMapper, self).__init__(logger, ["audio"])
        self.settings = None
        self.streams_by_language: Dict[str, List[AudioStreamInfo]] = {}
        self.stereo_streams_by_language: Dict[str, AudioStreamInfo] = {}

    def set_settings(self, settings: Settings) -> None:
        """Set plugin settings"""
        self.settings = settings

    def analyze_streams(self, probe_streams: List[dict]) -> None:
        """Analyze all audio streams and categorize them"""
        logger.debug("Beginning analysis of audio streams...")

        for stream in probe_streams:
            if stream.get("codec_type") != "audio":
                continue

            stream_info = AudioStreamInfo(stream)
            logger.info(f"""
    Found audio stream:
        Index: {stream_info.stream_index}
        Codec: {stream_info.codec_name}
        Channels: {stream_info.channels}
        Sample Rate: {stream_info.sample_rate}
        Language: {stream_info.language}
        Bitrate: {stream_info.bit_rate}
        Is Atmos: {stream_info.is_atmos}
        Title: {stream_info.title or 'None'}
    """)

            # Track streams by language
            if stream_info.language not in self.streams_by_language:
                self.streams_by_language[stream_info.language] = []
                logger.debug(f"First stream found for language: {stream_info.language}")
            self.streams_by_language[stream_info.language].append(stream_info)

            # Track existing stereo streams
            if stream_info.channels == 2:
                self.stereo_streams_by_language[stream_info.language] = stream_info
                logger.debug(
                    f"Found existing stereo stream for language: {stream_info.language}"
                )

        logger.debug(f"Found {len(probe_streams)} total streams")
        logger.debug(f"Found {len(self.streams_by_language)} language(s)")
        logger.debug(
            f"Found {len(self.stereo_streams_by_language)} existing stereo stream(s)"
        )

    def test_stream_needs_processing(self, stream_info: dict) -> bool:
        """Determine if stream needs processing"""
        analysis = AudioStreamInfo(stream_info)

        logger.debug(f"""
    Testing stream for processing:
        Index: {analysis.stream_index}
        Codec: {analysis.codec_name}
        Channels: {analysis.channels}
        Sample Rate: {analysis.sample_rate}
        Language: {analysis.language}
    """)

        # Always process if force enabled
        if self.settings.get_setting("force_processing"):
            logger.info("Stream needs processing: Force processing enabled")
            return True

        # Always process for normalization if enabled
        if self.settings.get_setting("enable_loudnorm"):
            logger.info("Stream needs processing: Loudness normalization enabled")
            return True

        # Process if sample rate needs conversion
        if analysis.sample_rate > self.settings.get_setting("target_sample_rate"):
            logger.info(
                f"Stream needs processing: Sample rate {analysis.sample_rate} exceeds target {self.settings.get_setting('target_sample_rate')}"
            )
            return True

        # Process if we need to create stereo version
        if (
            analysis.channels > 2
            and self.settings.get_setting("create_stereo_tracks")
            and analysis.language not in self.stereo_streams_by_language
        ):
            logger.info(
                f"Stream needs processing: Need to create stereo version for language {analysis.language}"
            )
            return True

        logger.debug("Stream does not need processing")
        return False

    def get_encoder_for_stream(self, stream_info: AudioStreamInfo) -> str:
        """Get appropriate encoder based on stream properties"""
        if stream_info.channels > 6:
            return self.settings.get_setting("multichannel_71_encoder")
        elif stream_info.channels > 2:
            return self.settings.get_setting("multichannel_51_encoder")
        return self.settings.get_setting("stereo_encoder")

    def calculate_bitrate(
        self, stream_info: AudioStreamInfo, target_channels: int = None
    ) -> int:
        """Calculate appropriate bitrate based on channels"""
        if target_channels is None:
            target_channels = stream_info.channels

        try:
            # Base on source bitrate if available
            per_channel = int(stream_info.bit_rate / stream_info.channels)
            return per_channel * target_channels
        except (ZeroDivisionError, ValueError):
            # Fallback to 64k per channel
            return 64000 * target_channels

    def get_loudnorm_filter(self, is_stereo: bool = False) -> str:
        """Get loudnorm filter string based on channel count"""
        if not self.settings.get_setting("enable_loudnorm"):
            return None

        if is_stereo:
            return (
                f"loudnorm=I={self.settings.get_setting('stereo_norm_i')}:"
                f"LRA={self.settings.get_setting('stereo_norm_lra')}:"
                f"TP={self.settings.get_setting('stereo_norm_tp')}"
            )
        else:
            return (
                f"loudnorm=I={self.settings.get_setting('surround_norm_i')}:"
                f"LRA={self.settings.get_setting('surround_norm_lra')}:"
                f"TP={self.settings.get_setting('surround_norm_tp')}"
            )

    def build_stream_mapping(self, stream_info: dict, stream_id: int):
        """Build FFmpeg mapping for stream processing"""
        analysis = AudioStreamInfo(stream_info)
        logger.info(f"""
    Building stream mapping for:
        Stream ID: {stream_id}
        Index: {analysis.stream_index}
        Codec: {analysis.codec_name}
        Channels: {analysis.channels}
        Sample Rate: {analysis.sample_rate}
        Language: {analysis.language}
    """)

        stream_mapping = []
        stream_encoding = []

        # Map original/main stream
        stream_mapping.extend(["-map", f"0:a:{stream_id}"])
        logger.debug(f"Mapped original stream: 0:a:{stream_id}")

        # Handle main stream processing
        needs_conversion = analysis.sample_rate > self.settings.get_setting(
            "target_sample_rate"
        )
        if needs_conversion:
            # Choose encoder based on channel count
            if analysis.channels > 6:
                encoder = self.settings.get_setting("multichannel_71_encoder")
                logger.info(f"Using 7.1 encoder: {encoder}")
            else:
                encoder = self.settings.get_setting("multichannel_51_encoder")
                logger.info(f"Using 5.1/stereo encoder: {encoder}")

            stream_encoding.extend(
                [
                    f"-c:a:{stream_id}",
                    encoder,
                    f"-ar:a:{stream_id}",
                    str(self.settings.get_setting("target_sample_rate")),
                ]
            )
            logger.debug(
                f"Converting sample rate to: {self.settings.get_setting('target_sample_rate')}"
            )
        else:
            # Copy original codec but still apply normalization
            stream_encoding.extend([f"-c:a:{stream_id}", "copy"])
            logger.debug("Copying original stream codec")

        # Apply main stream normalization
        if self.settings.get_setting("enable_loudnorm"):
            loudnorm = self.get_loudnorm_filter(is_stereo=False)
            if loudnorm:
                stream_encoding.extend([f"-filter:a:{stream_id}", loudnorm])
                logger.debug(f"Applied surround normalization: {loudnorm}")

        # Create stereo version if needed
        if (
            analysis.channels > 2
            and self.settings.get_setting("create_stereo_tracks")
            and analysis.language not in self.stereo_streams_by_language
        ):
            next_stream_id = len(self.streams_by_language)
            stereo_encoder = self.settings.get_setting("stereo_encoder")
            stereo_bitrate = self.calculate_bitrate(analysis, target_channels=2)

            logger.info(f"""
    Creating stereo version:
        Original Stream ID: {stream_id}
        New Stream ID: {next_stream_id}
        Encoder: {stereo_encoder}
        Bitrate: {stereo_bitrate}k
    """)

            # Map same input stream again
            stream_mapping.extend(["-map", f"0:a:{stream_id}"])
            stream_encoding.extend(
                [
                    f"-c:a:{next_stream_id}",
                    stereo_encoder,
                    f"-ac:a:{next_stream_id}",
                    "2",
                    f"-ar:a:{next_stream_id}",
                    str(self.settings.get_setting("target_sample_rate")),
                    f"-b:a:{next_stream_id}",
                    f"{stereo_bitrate}k",
                    f"-metadata:s:a:{next_stream_id}",
                    f'title={self.settings.get_setting("stereo_track_suffix")}',
                ]
            )

            # Apply stereo normalization
            stereo_loudnorm = self.get_loudnorm_filter(is_stereo=True)
            if stereo_loudnorm:
                stream_encoding.extend([f"-filter:a:{next_stream_id}", stereo_loudnorm])
                logger.debug(f"Applied stereo normalization: {stereo_loudnorm}")

        logger.debug("Final stream mapping: " + " ".join(stream_mapping))
        logger.debug("Final stream encoding: " + " ".join(stream_encoding))

        return {"stream_mapping": stream_mapping, "stream_encoding": stream_encoding}


def on_library_management_file_test(data):
    """
    Runner function - enables additional actions during the library management file tests.
    """
    # Get the path to the file
    abspath = data.get("path")

    # Get file probe
    probe = Probe(logger, allowed_mimetypes=["video"])
    if not probe.file(abspath):
        # File probe failed, skip the rest of this test
        return data

    # Configure settings object
    settings = Settings(library_id=data.get("library_id"))

    # Get stream mapper
    mapper = PluginStreamMapper()
    mapper.set_settings(settings)
    mapper.set_probe(probe)

    # Analyze all streams
    probe_streams = probe.get_probe()["streams"]
    mapper.analyze_streams(probe_streams)

    # Check if any streams need processing
    if mapper.streams_need_processing():
        data["add_file_to_pending_tasks"] = True
        logger.debug(
            f"File '{abspath}' should be added to task list - streams require processing"
        )
    else:
        logger.debug(f"File '{abspath}' does not require processing")

    return data


def on_worker_process(data):
    """
    Runner function - enables additional configured processing jobs during the worker stages of a task.
    """
    # Default to no FFMPEG command required
    data["exec_command"] = []
    data["repeat"] = False

    # Get the path to the file
    abspath = data.get("file_in")

    # Get file probe
    probe = Probe(logger, allowed_mimetypes=["video"])
    if not probe.file(abspath):
        # File probe failed, skip the rest of this test
        return data

    # Configure settings object
    settings = Settings(library_id=data.get("library_id"))

    # Get stream mapper
    mapper = PluginStreamMapper()
    mapper.set_settings(settings)
    mapper.set_probe(probe)

    # Analyze streams
    probe_streams = probe.get_probe()["streams"]
    mapper.analyze_streams(probe_streams)

    if mapper.streams_need_processing():
        # Set the input file
        mapper.set_input_file(abspath)

        # Set the output file
        mapper.set_output_file(data.get("file_out"))

        # Get generated ffmpeg args
        ffmpeg_args = mapper.get_ffmpeg_args()

        # Add container options
        ffmpeg_args += [
            # Copy video streams
            "-map",
            "0:v?",
            "-c:v",
            "copy",
            # Copy subtitle streams
            "-map",
            "0:s?",
            "-c:s",
            "copy",
            # Copy chapters
            "-map_chapters",
            "0",
            # Set max muxing queue size
            "-max_muxing_queue_size",
            str(settings.get_setting("max_muxing_queue_size")),
        ]

        # Apply ffmpeg args to command
        data["exec_command"] = ["ffmpeg"]
        data["exec_command"] += ffmpeg_args

        # Set the parser
        parser = Parser(logger)
        parser.set_probe(probe)
        data["command_progress_parser"] = parser.parse_progress

        logger.debug("FFmpeg command: {}".format(" ".join(data["exec_command"])))

    return data


def on_postprocessor_task_results(data):
    """
    Runner function - provides a means for additional postprocessor functions based on the task success.

    The 'data' object argument includes:
        task_processing_success         - Boolean, did all task processes complete successfully.
        file_move_processes_success     - Boolean, did all postprocessor movement tasks complete successfully.
        destination_files               - List containing all file paths created by postprocessor file movements.
        source_data                     - Dictionary containing data pertaining to the original source file.
    """
    # Only proceed if task was successful
    if not data.get("task_processing_success"):
        logger.debug("Task was not successful, skipping post-processing")
        return data

    # Configure settings object
    settings = Settings(library_id=data.get("library_id"))

    # Store processing history
    for destination_file in data.get("destination_files", []):
        try:
            directory_info = UnmanicDirectoryInfo(os.path.dirname(destination_file))

            # Store processing info
            info = {
                "version": "1.0",
                "settings": {
                    "target_sample_rate": settings.get_setting("target_sample_rate"),
                    "stereo_encoder": settings.get_setting("stereo_encoder"),
                    "loudnorm_enabled": settings.get_setting("enable_loudnorm"),
                    "surround_norm_i": settings.get_setting("surround_norm_i"),
                    "stereo_norm_i": settings.get_setting("stereo_norm_i"),
                },
            }

            directory_info.set("seb_custom", os.path.basename(destination_file), info)
            directory_info.save()

            logger.debug(f"Stored processing info for file: {destination_file}")

        except Exception as e:
            logger.error(f"Failed to store processing info: {str(e)}")

    return data


def test_logger():
    """Helper function to test plugin logging"""
    logger.debug("Audio Processor plugin test debug message")
    logger.info("Audio Processor plugin test info message")
    logger.warning("Audio Processor plugin test warning message")
    logger.error("Audio Processor plugin test error message")
    logger.critical("Audio Processor plugin test critical message")
