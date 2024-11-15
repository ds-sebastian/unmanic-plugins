#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
unmanic-plugins.plugin.py

Written by:               Claude <claude@anthropic.com>
Date:                     14 Nov 2024

Copyright:
    Copyright (C) 2024 Claude/Anthropic

    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General
    Public License as published by the Free Software Foundation, version 3.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License along with this program.
    If not, see <https://www.gnu.org/licenses/>.

"""

import logging

# Import custom ffmpeg helpers
from seb_custom.lib.ffmpeg import Parser, Probe, StreamMapper
from unmanic.libs.unplugins.settings import PluginSettings

# Configure plugin logger
logger = logging.getLogger("Unmanic.Plugin.seb_custom")


class Settings(PluginSettings):
    settings = {
        "force_processing": False,
        "target_sample_rate": "48000",
        "max_muxing_queue_size": 2048,
    }

    def __init__(self, *args, **kwargs):
        super(Settings, self).__init__(*args, **kwargs)
        self.form_settings = {
            "force_processing": {
                "label": "Force processing of all audio streams regardless of sample rate"
            },
            "target_sample_rate": {
                "label": "Target sample rate (Hz)",
                "input_type": "slider",
                "slider_options": {
                    "min": 8000,
                    "max": 96000,
                    "step": 100,
                },
            },
            "max_muxing_queue_size": {
                "label": "Max input stream packet buffer",
                "input_type": "slider",
                "slider_options": {
                    "min": 1024,
                    "max": 10240,
                },
            },
        }


class PluginStreamMapper(StreamMapper):
    def __init__(self):
        super(PluginStreamMapper, self).__init__(logger, ["audio"])
        self.settings = None
        self.target_sample_rate = 48000  # Default

    def set_settings(self, settings):
        self.settings = settings
        try:
            self.target_sample_rate = int(
                self.settings.get_setting("target_sample_rate")
            )
        except:
            logger.warning("Failed to parse target sample rate, using default 48000")
            self.target_sample_rate = 48000

    def test_stream_needs_processing(self, stream_info: dict):
        """
        Determine if this stream needs to be processed.
        """
        if not stream_info.get("sample_rate"):
            logger.debug("No sample rate found in stream info")
            return False

        try:
            current_sample_rate = int(stream_info.get("sample_rate"))
            logger.debug(f"Stream sample rate: {current_sample_rate}Hz")
            logger.debug(f"Target sample rate: {self.target_sample_rate}Hz")

            # Check if we need to process based on sample rate
            needs_processing = current_sample_rate > self.target_sample_rate

            if needs_processing:
                logger.debug(
                    f"Stream needs processing: {current_sample_rate}Hz > {self.target_sample_rate}Hz"
                )
            else:
                logger.debug(
                    f"Stream does not need processing: {current_sample_rate}Hz <= {self.target_sample_rate}Hz"
                )

            # Check force processing flag
            if self.settings.get_setting("force_processing"):
                logger.debug(
                    "Force processing enabled - will process stream regardless of sample rate"
                )
                return True

            return needs_processing

        except Exception as e:
            logger.error(f"Error checking stream sample rate: {e}")
            return False

    def custom_stream_mapping(self, stream_info: dict, stream_id: int):
        """
        Generate the custom mapping for this stream
        """
        current_sample_rate = int(stream_info.get("sample_rate", 0))
        channel_count = int(stream_info.get("channels", 2))

        # Select encoder based on channel count
        if channel_count > 6:
            encoder = "libopus"
            logger.debug(f"Using OPUS encoder for {channel_count} channels")
        else:
            encoder = "libfdk_aac"
            logger.debug(f"Using AAC encoder for {channel_count} channels")

        stream_encoding = [
            "-c:a:{}".format(stream_id),
            encoder,
        ]

        # Add sample rate conversion if needed
        if current_sample_rate > self.target_sample_rate:
            stream_encoding += [
                "-ar:a:{}".format(stream_id),
                str(self.target_sample_rate),
            ]
            logger.debug(
                f"Adding sample rate conversion from {current_sample_rate}Hz to {self.target_sample_rate}Hz"
            )

        # Calculate bitrate (64k per channel)
        bitrate = channel_count * 64
        stream_encoding += ["-b:a:{}".format(stream_id), "{}k".format(bitrate)]
        logger.debug(f"Setting bitrate to {bitrate}k for {channel_count} channels")

        return {
            "stream_mapping": ["-map", "0:a:{}".format(stream_id)],
            "stream_encoding": stream_encoding,
        }


def on_library_management_file_test(data):
    """
    Runner function - enables additional actions during the library management file tests.

    :param data: The data object
    :return: The data object
    """
    # Get the path to the file
    abspath = data.get("path")

    # Get file probe
    probe = Probe(logger, allowed_mimetypes=["audio", "video"])
    if not probe.file(abspath):
        # File probe failed, skip the rest of this test
        return data

    # Configure settings object
    settings = Settings(library_id=data.get("library_id"))

    # Get stream mapper
    mapper = PluginStreamMapper()
    mapper.set_settings(settings)
    mapper.set_probe(probe)

    if mapper.streams_need_processing():
        # Mark this file to be added to the pending tasks
        data["add_file_to_pending_tasks"] = True
        logger.debug(
            f"File '{abspath}' should be added to task list. Found streams requiring sample rate conversion."
        )
    else:
        logger.debug(
            f"File '{abspath}' does not contain streams requiring sample rate conversion."
        )

    return data


def on_worker_process(data):
    """
    Runner function - enables additional configured processing jobs during the worker stages of a task.

    :param data: The data object
    :return: The data object
    """
    # Default to no FFMPEG command required
    data["exec_command"] = []
    data["repeat"] = False

    # Get the path to the file
    abspath = data.get("file_in")

    # Get file probe
    probe = Probe(logger, allowed_mimetypes=["audio", "video"])
    if not probe.file(abspath):
        # File probe failed, skip the rest of this test
        return data

    # Configure settings object
    settings = Settings(library_id=data.get("library_id"))

    # Get stream mapper
    mapper = PluginStreamMapper()
    mapper.set_settings(settings)
    mapper.set_probe(probe)

    if mapper.streams_need_processing():
        # Set the input file
        mapper.set_input_file(abspath)

        # Set the output file
        mapper.set_output_file(data.get("file_out"))

        # Get generated ffmpeg args
        ffmpeg_args = mapper.get_ffmpeg_args()

        if ffmpeg_args:
            # Apply ffmpeg args to command
            data["exec_command"] = ["ffmpeg"]
            data["exec_command"] += ffmpeg_args

            # Set the parser
            parser = Parser(logger)
            parser.set_probe(probe)
            data["command_progress_parser"] = parser.parse_progress

            logger.debug("FFmpeg Command: {}".format(" ".join(data["exec_command"])))
        else:
            logger.debug("No ffmpeg args generated - stream may not need processing")

    return data
