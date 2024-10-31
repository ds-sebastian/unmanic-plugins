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

from lib.ffmpeg import Parser, Probe, StreamMapper
from unmanic.libs.directoryinfo import UnmanicDirectoryInfo
from unmanic.libs.unplugins.settings import PluginSettings

# Configure plugin logger
logger = logging.getLogger("Unmanic.Plugin.seb_custom")


class Settings(PluginSettings):
    """Plugin settings and UI configuration"""

    settings = {
        # General processing options
        "keep_original_streams": True,
        "force_processing": False,
        # Sample rate handling
        "target_sample_rate": 48000,
        "handle_atmos": True,
        "preserve_atmos_over48k": False,
        # Audio encoding
        "stereo_encoder": "libfdk_aac",  # or "opus"
        "multichannel_71_encoder": "opus",
        "multichannel_51_encoder": "libfdk_aac",
        # Stereo downmix
        "create_stereo_tracks": True,
        "stereo_track_suffix": "AAC Stereo",
        # Loudness normalization
        "enable_loudnorm": True,
        "surround_norm_i": "-24.0",
        "surround_norm_lra": "7.0",
        "surround_norm_tp": "-2.0",
        "stereo_norm_i": "-16.0",
        "stereo_norm_lra": "11.0",
        "stereo_norm_tp": "-1.0",
        # Advanced
        "max_muxing_queue_size": 2048,
    }

    def __init__(self, *args, **kwargs):
        super(Settings, self).__init__(*args, **kwargs)
        self.form_settings = {
            "keep_original_streams": {
                "label": "Keep original audio streams",
                "description": "Preserve original streams alongside processed versions",
            },
            "force_processing": {
                "label": "Force processing of all streams",
                "description": "Process streams even if they meet requirements",
            },
            "target_sample_rate": {
                "label": "Target sample rate",
                "input_type": "select",
                "select_options": [
                    {"value": "48000", "label": "48 kHz"},
                    {"value": "96000", "label": "96 kHz"},
                ],
            },
            "handle_atmos": {
                "label": "Handle Atmos/TrueHD content",
                "description": "Enable special handling of Atmos/TrueHD streams",
            },
            "preserve_atmos_over48k": {
                "label": "Preserve high sample rate Atmos",
                "description": "Keep Atmos streams even above target sample rate",
            },
            "stereo_encoder": {
                "label": "Stereo encoder",
                "input_type": "select",
                "select_options": [
                    {"value": "libfdk_aac", "label": "libfdk_aac (Better Quality)"},
                    {"value": "opus", "label": "Opus (Better Compatibility)"},
                ],
            },
            "multichannel_71_encoder": {
                "label": "7.1 Channel encoder",
                "input_type": "select",
                "select_options": [
                    {"value": "opus", "label": "Opus (Recommended)"},
                    {"value": "libfdk_aac", "label": "libfdk_aac"},
                ],
            },
            "create_stereo_tracks": {
                "label": "Create stereo downmix tracks",
                "description": "Create stereo versions of multichannel audio",
            },
            "enable_loudnorm": {
                "label": "Enable loudness normalization",
                "description": "Apply professional loudness standards",
            },
            "surround_norm_i": {
                "label": "Surround Integrated Loudness",
                "input_type": "slider",
                "slider_options": {"min": -70.0, "max": -5.0, "step": 0.1},
            },
            "stereo_norm_i": {
                "label": "Stereo Integrated Loudness",
                "input_type": "slider",
                "slider_options": {"min": -70.0, "max": -5.0, "step": 0.1},
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
        for stream in probe_streams:
            if stream.get("codec_type") != "audio":
                continue

            stream_info = AudioStreamInfo(stream)

            # Track streams by language
            if stream_info.language not in self.streams_by_language:
                self.streams_by_language[stream_info.language] = []
            self.streams_by_language[stream_info.language].append(stream_info)

            # Track existing stereo streams
            if stream_info.channels == 2:
                self.stereo_streams_by_language[stream_info.language] = stream_info

    def test_stream_needs_processing(self, stream_info: dict) -> bool:
        """Determine if stream needs processing"""
        analysis = AudioStreamInfo(stream_info)

        # Force processing if enabled
        if self.settings.get_setting("force_processing"):
            return True

        # Check sample rate
        if analysis.sample_rate > self.settings.get_setting("target_sample_rate"):
            # Special handling for Atmos
            if analysis.is_atmos and self.settings.get_setting(
                "preserve_atmos_over48k"
            ):
                return False
            return True

        # Check for stereo conversion needs
        if (
            analysis.channels > 2
            and self.settings.get_setting("create_stereo_tracks")
            and analysis.language not in self.stereo_streams_by_language
        ):
            return True

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
        encoder = self.get_encoder_for_stream(analysis)

        stream_mapping = []
        stream_encoding = []

        # Map original stream
        stream_mapping.extend(["-map", f"0:a:{stream_id}"])

        if self.test_stream_needs_processing(stream_info):
            # Convert sample rate if needed
            if analysis.sample_rate > self.settings.get_setting("target_sample_rate"):
                stream_encoding.extend(
                    [
                        f"-ar:a:{stream_id}",
                        str(self.settings.get_setting("target_sample_rate")),
                    ]
                )

            # Apply encoder and bitrate
            bitrate = self.calculate_bitrate(analysis)
            stream_encoding.extend(
                [f"-c:a:{stream_id}", encoder, f"-b:a:{stream_id}", f"{bitrate}k"]
            )

            # Apply loudness normalization
            loudnorm = self.get_loudnorm_filter(is_stereo=analysis.channels <= 2)
            if loudnorm:
                stream_encoding.extend(["-filter:a", loudnorm])

        else:
            # Copy stream as-is
            stream_encoding.extend([f"-c:a:{stream_id}", "copy"])

        # Create stereo downmix if needed
        if (
            analysis.channels > 2
            and self.settings.get_setting("create_stereo_tracks")
            and analysis.language not in self.stereo_streams_by_language
        ):
            next_stream_id = len(self.streams_by_language)
            stereo_encoder = self.settings.get_setting("stereo_encoder")
            stereo_bitrate = self.calculate_bitrate(analysis, target_channels=2)

            stream_mapping.extend(["-map", f"0:a:{stream_id}"])
            stream_encoding.extend(
                [
                    f"-c:a:{next_stream_id}",
                    stereo_encoder,
                    f"-ac:a:{next_stream_id}",
                    "2",
                    f"-b:a:{next_stream_id}",
                    f"{stereo_bitrate}k",
                    f"-metadata:s:a:{next_stream_id}",
                    f'title={self.settings.get_setting("stereo_track_suffix")}',
                ]
            )

            # Apply stereo loudnorm
            stereo_loudnorm = self.get_loudnorm_filter(is_stereo=True)
            if stereo_loudnorm:
                stream_encoding.extend([f"-filter:a:{next_stream_id}", stereo_loudnorm])

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
