#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
unmanic-plugins.plugin.py

Written by:               Josh.5 <jsunnex@gmail.com>
Modified by:             Claude <claude@anthropic.com>
Date:                     14 Nov 2024

Copyright:
    Copyright (C) 2021 Josh Sunnex
    Copyright (C) 2024 Anthropic

    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General
    Public License as published by the Free Software Foundation, version 3.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License along with this program.
    If not, see <https://www.gnu.org/licenses/>.

"""

import logging

from seb_custom.lib.ffmpeg import Parser, Probe, StreamMapper
from unmanic.libs.unplugins.settings import PluginSettings

# Configure plugin logger
logger = logging.getLogger("Unmanic.Plugin.encoder_audio_aac")


class Settings(PluginSettings):
    settings = {
        "advanced": False,
        "force_processing": False,
        "target_sample_rate": "48000",
        "max_muxing_queue_size": 2048,
        "main_options": "",
        "advanced_options": "",
        "custom_options": "",
    }

    def __init__(self, *args, **kwargs):
        super(Settings, self).__init__(*args, **kwargs)
        self.form_settings = {
            "advanced": {
                "label": "Write your own FFmpeg params",
            },
            "force_processing": {
                "label": "Force processing (process streams even if audio is already encoded)",
            },
            "target_sample_rate": {
                "label": "Target sample rate",
                "input_type": "slider",
                "slider_options": {
                    "min": 8000,
                    "max": 96000,
                    "step": 100,
                },
            },
            "max_muxing_queue_size": self.__set_max_muxing_queue_size_form_settings(),
            "main_options": self.__set_main_options_form_settings(),
            "advanced_options": self.__set_advanced_options_form_settings(),
            "custom_options": self.__set_custom_options_form_settings(),
        }

    def __set_max_muxing_queue_size_form_settings(self):
        values = {
            "label": "Max input stream packet buffer",
            "input_type": "slider",
            "slider_options": {
                "min": 1024,
                "max": 10240,
            },
        }
        if self.get_setting("advanced"):
            values["display"] = "hidden"
        return values

    def __set_main_options_form_settings(self):
        values = {
            "label": "Write your own custom main options",
            "input_type": "textarea",
        }
        if not self.get_setting("advanced"):
            values["display"] = "hidden"
        return values

    def __set_advanced_options_form_settings(self):
        values = {
            "label": "Write your own custom advanced options",
            "input_type": "textarea",
        }
        if not self.get_setting("advanced"):
            values["display"] = "hidden"
        return values

    def __set_custom_options_form_settings(self):
        values = {
            "label": "Write your own custom audio options",
            "input_type": "textarea",
        }
        if not self.get_setting("advanced"):
            values["display"] = "hidden"
        return values


class PluginStreamMapper(StreamMapper):
    def __init__(self):
        super(PluginStreamMapper, self).__init__(logger, ["audio"])
        self.codec = "aac"
        self.encoder = "libfdk_aac"
        self.settings = None

    def set_default_values(self, settings, abspath, probe):
        """
        Configure the stream mapper with defaults

        :param settings:
        :param abspath:
        :param probe:
        :return:
        """
        self.abspath = abspath
        # Set the file probe data
        self.set_probe(probe)
        # Set the input file
        self.set_input_file(abspath)
        # Configure settings
        self.settings = settings

        # Build default options of advanced mode
        if self.settings.get_setting("advanced"):
            # If any main options are provided, overwrite them
            main_options = settings.get_setting("main_options").split()
            if main_options:
                # Overwrite all main options
                self.main_options = main_options
            # If any advanced options are provided, overwrite them
            advanced_options = settings.get_setting("advanced_options").split()
            if advanced_options:
                # Overwrite all advanced options
                self.advanced_options = advanced_options

    @staticmethod
    def calculate_bitrate(stream_info: dict):
        channels = stream_info.get("channels", 2)
        return int(channels) * 64

    def test_stream_needs_processing(self, stream_info: dict):
        """
        Test if the stream needs to be processed.

        A stream needs processing if:
        - Its sample rate is higher than target_sample_rate (regardless of codec)
        - OR if force_processing is enabled and it's not already encoded with libfdk_aac
        """
        force_processing = self.settings.get_setting("force_processing")
        target_sample_rate = int(self.settings.get_setting("target_sample_rate"))

        # Get current sample rate
        try:
            current_sample_rate = int(stream_info.get("sample_rate", 0))
        except (TypeError, ValueError):
            logger.warning("Unable to determine sample rate for stream")
            return False

        # Check if sample rate exceeds target
        needs_sample_rate_conversion = current_sample_rate > target_sample_rate

        # Log the stream details
        logger.debug(f"Stream codec: {stream_info.get('codec_name')}")
        logger.debug(
            f"Stream sample rate: {current_sample_rate}Hz, Target: {target_sample_rate}Hz"
        )

        if needs_sample_rate_conversion:
            logger.debug("Stream requires sample rate conversion regardless of codec")
            return True

        # If we don't need sample rate conversion, check if we should force process
        if force_processing:
            # Check if already encoded with libfdk_aac
            is_libfdk = (
                "tags" in stream_info
                and "ENCODER" in stream_info.get("tags")
                and self.encoder in stream_info.get("tags")["ENCODER"]
            )
            return not is_libfdk

        return False

    def custom_stream_mapping(self, stream_info: dict, stream_id: int):
        """
        Generate stream mapping including sample rate conversion if needed
        """
        stream_encoding = ["-c:a:{}".format(stream_id), self.encoder]

        # Check if sample rate conversion is needed
        current_sample_rate = int(stream_info.get("sample_rate", 0))
        target_sample_rate = int(self.settings.get_setting("target_sample_rate"))
        current_codec = stream_info.get("codec_name", "unknown")

        if current_sample_rate > target_sample_rate:
            # Add sample rate conversion
            stream_encoding += ["-ar:a:{}".format(stream_id), str(target_sample_rate)]
            logger.debug(
                f"Converting {current_codec} stream from {current_sample_rate}Hz to {target_sample_rate}Hz"
            )
        else:
            logger.debug(
                f"Processing {current_codec} stream due to force_processing (sample rate already {current_sample_rate}Hz)"
            )

        if self.settings.get_setting("advanced"):
            stream_encoding += self.settings.get_setting("custom_options").split()
        else:
            # Automatically detect bitrate for this stream
            if stream_info.get("channels"):
                # Use 64K for the bitrate per channel
                calculated_bitrate = self.calculate_bitrate(stream_info)
                channels = int(stream_info.get("channels"))
                if channels > 6:
                    channels = 6
                stream_encoding += [
                    "-ac:a:{}".format(stream_id),
                    "{}".format(channels),
                    "-b:a:{}".format(stream_id),
                    "{}k".format(calculated_bitrate),
                ]

        return {
            "stream_mapping": ["-map", "0:a:{}".format(stream_id)],
            "stream_encoding": stream_encoding,
        }


def on_library_management_file_test(data):
    """
    Runner function - enables additional actions during the library management file tests.

    The 'data' object argument includes:
        path                            - String containing the full path to the file being tested.
        issues                          - List of currently found issues for not processing the file.
        add_file_to_pending_tasks       - Boolean, is the file currently marked to be added to the queue for processing.

    :param data:
    :return:
    """
    # Get the path to the file
    abspath = data.get("path")

    # Get file probe
    probe = Probe(logger, allowed_mimetypes=["audio", "video"])
    if not probe.file(abspath):
        # File probe failed, skip the rest of this test
        return data

    # Configure settings object (maintain compatibility with v1 plugins)
    if data.get("library_id"):
        settings = Settings(library_id=data.get("library_id"))
    else:
        settings = Settings()

    # Get stream mapper
    mapper = PluginStreamMapper()
    mapper.set_default_values(settings, abspath, probe)

    if mapper.streams_need_processing():
        # Mark this file to be added to the pending tasks
        data["add_file_to_pending_tasks"] = True
        logger.debug(
            "File '{}' needs processing - Found streams with sample rate > {}Hz".format(
                abspath, settings.get_setting("target_sample_rate")
            )
        )
    else:
        logger.debug(
            "File '{}' does not need processing - All streams at or below {}Hz".format(
                abspath, settings.get_setting("target_sample_rate")
            )
        )

    return data


def on_worker_process(data):
    """
    Runner function - enables additional configured processing jobs during the worker stages of a task.

    The 'data' object argument includes:
        exec_command            - A command that Unmanic should execute. Can be empty.
        command_progress_parser - A function that Unmanic can use to parse the STDOUT of the command to collect progress stats. Can be empty.
        file_in                - The source file to be processed by the command.
        file_out                - The destination that the command should output (may be the same as the file_in if necessary).
        original_file_path      - The absolute path to the original file.
        repeat                  - Boolean, should this runner be executed again once completed with the same variables.

    :param data:
    :return:
    """
    # Default to no FFMPEG command required. This prevents the FFMPEG command from running if it is not required
    data["exec_command"] = []
    data["repeat"] = False

    # Get the path to the file
    abspath = data.get("file_in")

    # Get file probe
    probe = Probe(logger, allowed_mimetypes=["audio", "video"])
    if not probe.file(abspath):
        # File probe failed, skip the rest of this test
        return data

    # Configure settings object (maintain compatibility with v1 plugins)
    settings = Settings(library_id=data.get("library_id"))

    # Get stream mapper
    mapper = PluginStreamMapper()
    mapper.set_default_values(settings, abspath, probe)

    if mapper.streams_need_processing():
        # Set the input file
        mapper.set_input_file(abspath)

        # Set the output file
        mapper.set_output_file(data.get("file_out"))

        # Get generated ffmpeg args
        ffmpeg_args = mapper.get_ffmpeg_args()

        # Apply ffmpeg args to command
        data["exec_command"] = ["ffmpeg"]
        data["exec_command"] += ffmpeg_args

        # Set the parser
        parser = Parser(logger)
        parser.set_probe(probe)
        data["command_progress_parser"] = parser.parse_progress

    return data
