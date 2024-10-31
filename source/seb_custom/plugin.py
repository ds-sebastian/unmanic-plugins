#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plugins.__init__.py

Written by:               Josh.5 <jsunnex@gmail.com>
Date:                     23 Aug 2021, (20:38 PM)

Copyright:
    Copyright (C) 2021 Josh Sunnex

    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General
    Public License as published by the Free Software Foundation, version 3.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License along with this program.
    If not, see <https://www.gnu.org/licenses/>.

"""

import logging

from lib.ffmpeg import Parser, Probe, StreamMapper
from unmanic.libs.unplugins.settings import PluginSettings

# Configure plugin logger
logger = logging.getLogger("Unmanic.Plugin.seb_custom")


class Settings(PluginSettings):
    settings = {
        # General settings
        "keep_original_streams": True,
        "force_processing": False,
        # Audio settings
        "target_sample_rate": 48000,
        "always_create_stereo": True,
        "stereo_encoder": "libfdk_aac",  # or "aac"
        # Loudness normalization
        "enable_loudnorm": True,
        "surround_I": "-24.0",
        "surround_LRA": "7.0",
        "surround_TP": "-2.0",
        "stereo_I": "-16.0",
        "stereo_LRA": "11.0",
        "stereo_TP": "-1.0",
        # Advanced settings
        "max_muxing_queue_size": 2048,
    }

    def __init__(self, *args, **kwargs):
        super(Settings, self).__init__(*args, **kwargs)
        self.form_settings = {
            "keep_original_streams": {
                "label": "Keep original audio streams",
            },
            "force_processing": {
                "label": "Force processing of all audio streams regardless of current format",
            },
            "target_sample_rate": {
                "label": "Target sample rate",
                "input_type": "select",
                "select_options": {
                    48000: "48kHz",
                    44100: "44.1kHz",
                },
            },
            "always_create_stereo": {
                "label": "Always create stereo version",
            },
            "stereo_encoder": {
                "label": "Stereo encoder",
                "input_type": "select",
                "select_options": {
                    "libfdk_aac": "libfdk_aac (better quality)",
                    "aac": "Native AAC (wider compatibility)",
                },
            },
            "enable_loudnorm": {
                "label": "Enable loudness normalization",
            },
            # Surround normalization settings
            "surround_I": {
                "label": "Surround - Integrated loudness target",
                "input_type": "slider",
                "slider_options": {
                    "min": -70.0,
                    "max": -5.0,
                    "step": 0.1,
                },
            },
            "surround_LRA": {
                "label": "Surround - Loudness range",
                "input_type": "slider",
                "slider_options": {
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1,
                },
            },
            "surround_TP": {
                "label": "Surround - Maximum true peak",
                "input_type": "slider",
                "slider_options": {
                    "min": -9.0,
                    "max": 0,
                    "step": 0.1,
                },
            },
            # Stereo normalization settings
            "stereo_I": {
                "label": "Stereo - Integrated loudness target",
                "input_type": "slider",
                "slider_options": {
                    "min": -70.0,
                    "max": -5.0,
                    "step": 0.1,
                },
            },
            "stereo_LRA": {
                "label": "Stereo - Loudness range",
                "input_type": "slider",
                "slider_options": {
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1,
                },
            },
            "stereo_TP": {
                "label": "Stereo - Maximum true peak",
                "input_type": "slider",
                "slider_options": {
                    "min": -9.0,
                    "max": 0,
                    "step": 0.1,
                },
            },
        }


class PluginStreamMapper(StreamMapper):
    def __init__(self):
        super(PluginStreamMapper, self).__init__(logger, ["audio"])
        self.settings = None
        self.existing_stereo_languages = set()

    def set_settings(self, settings):
        self.settings = settings

    def analyze_stream(self, stream_info: dict) -> dict:
        """Analyzes audio stream and returns relevant properties"""
        stream_data = {
            "channels": int(stream_info.get("channels", 2)),
            "sample_rate": int(stream_info.get("sample_rate", 48000)),
            "codec_name": stream_info.get("codec_name", "").lower(),
            "language": stream_info.get("tags", {}).get("language", "und"),
            "title": stream_info.get("tags", {}).get("title", ""),
            "bitrate": stream_info.get("bit_rate", "192k"),
            "is_atmos": "atmos" in stream_info.get("profile", "").lower()
            if "profile" in stream_info
            else False,
            "needs_sample_rate_conversion": int(stream_info.get("sample_rate", 48000))
            > self.settings.get_setting("target_sample_rate"),
        }

        logger.debug("Stream analysis: {}".format(stream_data))
        return stream_data

    def get_loudnorm_filter(self, is_stereo: bool) -> str:
        """Returns loudnorm filter string based on channel configuration"""
        if not self.settings.get_setting("enable_loudnorm"):
            return ""

        if is_stereo:
            return "loudnorm=I={}:LRA={}:TP={}".format(
                self.settings.get_setting("stereo_I"),
                self.settings.get_setting("stereo_LRA"),
                self.settings.get_setting("stereo_TP"),
            )
        else:
            return "loudnorm=I={}:LRA={}:TP={}".format(
                self.settings.get_setting("surround_I"),
                self.settings.get_setting("surround_LRA"),
                self.settings.get_setting("surround_TP"),
            )

    def calculate_stream_bitrate(self, stream_info: dict) -> str:
        """Calculates appropriate bitrate based on channel count"""
        try:
            original_bitrate = int(stream_info.get("bit_rate", "192000"))
            channels = int(stream_info.get("channels", 2))
            # Calculate per-channel bitrate and apply to target channels
            per_channel_bitrate = original_bitrate / (1000 * channels)
            target_bitrate = (
                int(per_channel_bitrate * 2)
                if channels > 2
                else int(per_channel_bitrate * channels)
            )
            return f"{target_bitrate}k"
        except (ValueError, TypeError):
            logger.debug("Failed to calculate bitrate, using default")
            return "128k"

    def test_stream_needs_processing(self, stream_info: dict) -> bool:
        """Determines if stream needs processing based on our criteria"""
        analysis = self.analyze_stream(stream_info)

        # Force processing if enabled
        if self.settings.get_setting("force_processing"):
            logger.debug("Force processing enabled - will process stream")
            return True

        # Check sample rate regardless of channel count
        if analysis["needs_sample_rate_conversion"]:
            logger.debug(
                "Sample rate {} exceeds target {} - needs processing".format(
                    analysis["sample_rate"],
                    self.settings.get_setting("target_sample_rate"),
                )
            )
            return True

        # Only create additional stereo version if:
        # 1. Original is >2 channels
        # 2. Always create stereo is enabled
        # 3. No stereo exists for this language
        if (
            analysis["channels"] > 2
            and self.settings.get_setting("always_create_stereo")
            and analysis["language"] not in self.existing_stereo_languages
        ):
            logger.debug(
                "Need to create stereo version for language: {}".format(
                    analysis["language"]
                )
            )
            return True

        return False

    def custom_stream_mapping(self, stream_info: dict, stream_id: int) -> dict:
        """Generates FFmpeg mapping for stream processing"""
        analysis = self.analyze_stream(stream_info)

        # Initialize stream mapping
        stream_mapping = {
            "stream_mapping": ["-map", "0:a:{}".format(stream_id)],
            "stream_encoding": [],
        }

        # Determine if we need to create a new stereo version
        creating_stereo = (
            analysis["channels"] > 2
            and self.settings.get_setting("always_create_stereo")
            and analysis["language"] not in self.existing_stereo_languages
        )

        # If it's already stereo/mono, just handle sample rate if needed
        if analysis["channels"] <= 2:
            if analysis["needs_sample_rate_conversion"]:
                logger.debug("Converting stereo/mono stream sample rate only")
                target_encoder = self.settings.get_setting("stereo_encoder")
                stream_encoding = [
                    "-c:a:{}".format(stream_id),
                    target_encoder,
                    "-ac:a:{}".format(stream_id),
                    str(analysis["channels"]),  # Keep original channel count
                    "-ar:a:{}".format(stream_id),
                    str(self.settings.get_setting("target_sample_rate")),
                ]

                # Add bitrate
                bitrate = self.calculate_stream_bitrate(stream_info)
                stream_encoding.extend(["-b:a:{}".format(stream_id), bitrate])

                # Add stereo loudnorm filter if enabled
                loudnorm = self.get_loudnorm_filter(True)
                if loudnorm:
                    stream_encoding.extend(["-filter:a:{}".format(stream_id), loudnorm])

                stream_mapping["stream_encoding"] = stream_encoding
                return stream_mapping
            else:
                logger.debug("Copying stereo/mono stream as-is")
                stream_mapping["stream_encoding"] = [
                    "-c:a:{}".format(stream_id),
                    "copy",
                ]
                return stream_mapping

        # Rest of the multichannel processing logic remains the same
        target_encoder = (
            self.settings.get_setting("stereo_encoder")
            if creating_stereo
            else "libfdk_aac"
        )
        target_channels = 2 if creating_stereo else min(analysis["channels"], 6)

        # Switch to OPUS for 7.1 content that needs sample rate conversion
        if analysis["channels"] > 6 and analysis["needs_sample_rate_conversion"]:
            target_encoder = "libopus"
            target_channels = analysis["channels"]

        # Build encoding parameters
        stream_encoding = [
            "-c:a:{}".format(stream_id),
            target_encoder,
            "-ac:a:{}".format(stream_id),
            str(target_channels),
            "-ar:a:{}".format(stream_id),
            str(self.settings.get_setting("target_sample_rate")),
        ]

        # Add bitrate
        bitrate = self.calculate_stream_bitrate(stream_info)
        stream_encoding.extend(["-b:a:{}".format(stream_id), bitrate])

        # Add loudnorm filter if enabled
        loudnorm = self.get_loudnorm_filter(creating_stereo)
        if loudnorm:
            stream_encoding.extend(["-filter:a:{}".format(stream_id), loudnorm])

        # Add metadata
        if creating_stereo:
            stream_encoding.extend(
                [
                    "-metadata:s:a:{}".format(stream_id),
                    "title=Stereo",
                    "-metadata:s:a:{}".format(stream_id),
                    "language={}".format(analysis["language"]),
                ]
            )

        stream_mapping["stream_encoding"] = stream_encoding
        logger.debug("Stream mapping for {}: {}".format(stream_id, stream_mapping))

        return stream_mapping


def on_library_management_file_test(data):
    """
    Runner function - enables additional actions during the library management file tests.
    """
    # Get the path to the file
    abspath = data.get("path")

    # Get file probe
    probe = Probe(logger, allowed_mimetypes=["audio", "video"])
    if not probe.file(abspath):
        # File probe failed, skip the rest of this test
        logger.debug("File probe failed for path: {}".format(abspath))
        return data

    # Configure settings object
    if data.get("library_id"):
        settings = Settings(library_id=data.get("library_id"))
    else:
        settings = Settings()

    # Get stream mapper
    mapper = PluginStreamMapper()
    mapper.set_settings(settings)
    mapper.set_probe(probe)

    # Analyze all audio streams first to identify existing stereo streams
    probe_streams = probe.get_probe()["streams"]
    for stream in probe_streams:
        if stream.get("codec_type") == "audio" and int(stream.get("channels", 0)) == 2:
            language = stream.get("tags", {}).get("language", "und")
            mapper.existing_stereo_languages.add(language)
            logger.debug(
                "Found existing stereo stream for language: {}".format(language)
            )

    # Check if any streams need processing
    streams_need_processing = False
    streams_to_process = []

    for stream in probe_streams:
        if stream.get("codec_type") == "audio":
            if mapper.test_stream_needs_processing(stream):
                streams_need_processing = True
                streams_to_process.append(
                    {
                        "index": stream.get("index"),
                        "language": stream.get("tags", {}).get("language", "und"),
                        "channels": stream.get("channels", 2),
                        "sample_rate": stream.get("sample_rate", 48000),
                    }
                )

    if streams_need_processing:
        # Mark this file to be added to the pending tasks
        data["add_file_to_pending_tasks"] = True
        logger.debug(
            "File '{}' needs processing. Streams to process: {}".format(
                abspath, streams_to_process
            )
        )
    else:
        logger.debug("File '{}' does not need processing".format(abspath))

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
    outpath = data.get("file_out")

    # Get file probe
    probe = Probe(logger, allowed_mimetypes=["audio", "video"])
    if not probe.file(abspath):
        logger.debug("Probe failed for file: {}".format(abspath))
        return data

    # Configure settings object
    if data.get("library_id"):
        settings = Settings(library_id=data.get("library_id"))
    else:
        settings = Settings()

    # Get stream mapper
    mapper = PluginStreamMapper()
    mapper.set_settings(settings)
    mapper.set_probe(probe)

    # Pre-scan for existing stereo streams
    probe_streams = probe.get_probe()["streams"]
    for stream in probe_streams:
        if stream.get("codec_type") == "audio" and int(stream.get("channels", 0)) == 2:
            language = stream.get("tags", {}).get("language", "und")
            mapper.existing_stereo_languages.add(language)

    # Generate FFmpeg args
    ffmpeg_args = [
        "-hide_banner",
        "-loglevel",
        "info",
        "-i",
        str(abspath),
        "-max_muxing_queue_size",
        str(settings.get_setting("max_muxing_queue_size")),
        "-map",
        "0:v",  # Map all video streams
        "-c:v",
        "copy",  # Copy video streams
    ]

    # Track audio stream count for indexing
    audio_stream_index = 0
    streams_processed = set()

    # First pass - handle original streams
    for stream in probe_streams:
        if stream.get("codec_type") == "audio":
            stream_id = str(audio_stream_index)
            needs_processing = mapper.test_stream_needs_processing(stream)

            if needs_processing or settings.get_setting("force_processing"):
                # Get stream mapping for processing
                mapping = mapper.custom_stream_mapping(stream, audio_stream_index)
                ffmpeg_args.extend(mapping["stream_mapping"])
                ffmpeg_args.extend(mapping["stream_encoding"])
                streams_processed.add(stream.get("index"))
                logger.debug(
                    "Processing audio stream {} with settings: {}".format(
                        audio_stream_index, mapping["stream_encoding"]
                    )
                )
            elif settings.get_setting("keep_original_streams"):
                # Copy original stream without processing
                ffmpeg_args.extend(
                    [
                        "-map",
                        "0:a:{}".format(stream_id),
                        "-c:a:{}".format(stream_id),
                        "copy",
                    ]
                )
                logger.debug(
                    "Copying original audio stream {} without processing".format(
                        audio_stream_index
                    )
                )

            audio_stream_index += 1

    # Second pass - create stereo versions if needed
    if settings.get_setting("always_create_stereo"):
        for stream in probe_streams:
            if (
                stream.get("codec_type") == "audio"
                and int(stream.get("channels", 0)) > 2
                and stream.get("tags", {}).get("language", "und")
                not in mapper.existing_stereo_languages
            ):
                # Create stereo version
                mapping = mapper.custom_stream_mapping(stream, audio_stream_index)
                ffmpeg_args.extend(mapping["stream_mapping"])
                ffmpeg_args.extend(mapping["stream_encoding"])
                logger.debug(
                    "Creating stereo version for stream {} as stream {}".format(
                        stream.get("index"), audio_stream_index
                    )
                )

                # Update tracking
                mapper.existing_stereo_languages.add(
                    stream.get("tags", {}).get("language", "und")
                )
                audio_stream_index += 1

    # Add subtitle mapping
    ffmpeg_args.extend(["-map", "0:s?", "-c:s", "copy"])

    # Add chapter mapping
    ffmpeg_args.extend(["-map_chapters", "0"])

    # Add output file
    ffmpeg_args.extend(["-y", str(outpath)])

    logger.debug("Complete FFmpeg command: {}".format(ffmpeg_args))

    # Apply FFmpeg args to command
    if audio_stream_index > 0:
        data["exec_command"] = ["ffmpeg"]
        data["exec_command"].extend(ffmpeg_args)

        # Set the parser
        parser = Parser(logger)
        parser.set_probe(probe)
        data["command_progress_parser"] = parser.parse_progress

    return data
