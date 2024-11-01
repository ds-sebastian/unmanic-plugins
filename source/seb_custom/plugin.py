#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
unmanic-plugins.plugin.py

Written by: Sebastian
Date: October 31, 2024

Plugin for managing audio streams in media files:
- Keeps original streams ≤48kHz untouched
- Re-encodes >48kHz streams using OPUS (>6ch) or libfdk_aac (≤6ch)
- Creates stereo compatibility versions of multichannel content
- Applies normalization during encoding
- Preserves all audio languages
"""

import logging
import os
import traceback
from datetime import datetime

from seb_custom.lib.ffmpeg import Parser, Probe, StreamMapper
from unmanic.libs.directoryinfo import UnmanicDirectoryInfo
from unmanic.libs.unplugins.settings import PluginSettings

# Configure plugin logger
logger = logging.getLogger("Unmanic.Plugin.seb_custom")


class Settings(PluginSettings):
    settings = {
        "keep_original_streams": True,
        "force_processing": False,
        "surround_normalize": {"I": "-24.0", "LRA": "7.0", "TP": "-2.0"},
        "stereo_normalize": {"I": "-16.0", "LRA": "11.0", "TP": "-1.0"},
    }


class PluginStreamMapper(StreamMapper):
    def __init__(self):
        super(PluginStreamMapper, self).__init__(logger, ["audio"])
        self.target_sample_rate = 48000
        self.stereo_versions = set()
        self.stream_count = 0
        self.settings = None
        logger.info("Initialized PluginStreamMapper")

    def set_settings(self, settings):
        self.settings = settings

    def analyze_stream(self, stream):
        """Analyze single audio stream and return processing details"""
        try:
            stream_info = {
                "index": stream.get("index", 0),
                "codec": stream.get("codec_name", "").lower(),
                "channels": int(stream.get("channels", 2)),
                "sample_rate": int(stream.get("sample_rate", "48000")),
                "bit_rate": stream.get("bit_rate"),
                "language": stream.get("tags", {}).get("language", "und"),
                "title": stream.get("tags", {}).get("title"),
                "disposition": stream.get("disposition", {}),
            }

            # Determine if processing needed
            needs_processing = False
            reasons = []

            # Check sample rate
            if stream_info["sample_rate"] > self.target_sample_rate:
                needs_processing = True
                reasons.append(
                    f"Sample rate {stream_info['sample_rate']} > {self.target_sample_rate}"
                )

            # Check if needs stereo version
            needs_stereo = False
            if stream_info["channels"] > 2:
                language_key = f"{stream_info['language']}_stereo"
                if language_key not in self.stereo_versions:
                    needs_stereo = True
                    self.stereo_versions.add(language_key)
                    reasons.append(
                        f"Needs stereo version for {stream_info['channels']} channels"
                    )

            # Determine encoder
            encoder = None
            if needs_processing:
                if stream_info["channels"] > 6:
                    encoder = "libopus"
                else:
                    encoder = "libfdk_aac"

            return {
                "info": stream_info,
                "needs_processing": needs_processing,
                "needs_stereo": needs_stereo,
                "encoder": encoder,
                "reasons": reasons,
            }

        except Exception as e:
            logger.error(f"Error analyzing stream: {str(e)}")
            return None

    def test_stream_needs_processing(self, stream_info: dict):
        """Override base method to implement our processing logic"""
        try:
            if stream_info.get("codec_type") != "audio":
                return False

            analysis = self.analyze_stream(stream_info)
            if not analysis:
                return False

            return analysis["needs_processing"] or analysis["needs_stereo"]

        except Exception as e:
            logger.error(f"Error in test_stream_needs_processing: {str(e)}")
            return False

    def custom_stream_mapping(self, stream_info: dict, stream_id: int):
        """Override base method to implement our custom mapping"""
        try:
            analysis = self.analyze_stream(stream_info)
            if not analysis:
                return None

            args = {
                "stream_mapping": ["-map", f"0:a:{stream_id}"],
                "stream_encoding": [],
            }

            # Handle main stream
            if analysis["needs_processing"]:
                args["stream_encoding"] = [
                    f"-c:a:{self.stream_count}",
                    analysis["encoder"],
                    f"-ar:a:{self.stream_count}",
                    str(self.target_sample_rate),
                ]

                # Add normalization filter
                if analysis["info"]["channels"] > 2:
                    norm = self.settings.settings["surround_normalize"]
                else:
                    norm = self.settings.settings["stereo_normalize"]

                args["stream_encoding"].extend(
                    [
                        f"-filter:a:{self.stream_count}",
                        f'loudnorm=I={norm["I"]}:LRA={norm["LRA"]}:TP={norm["TP"]}',
                    ]
                )
            else:
                args["stream_encoding"] = [f"-c:a:{self.stream_count}", "copy"]

            # Add metadata
            language = analysis["info"]["language"]
            if language != "und":
                args["stream_encoding"].extend(
                    [f"-metadata:s:a:{self.stream_count}", f"language={language}"]
                )

            title = analysis["info"]["title"]
            if title:
                args["stream_encoding"].extend(
                    [f"-metadata:s:a:{self.stream_count}", f"title={title}"]
                )

            # Increment stream counter
            self.stream_count += 1

            # Add stereo version if needed
            if analysis["needs_stereo"]:
                stereo_args = {
                    "stream_mapping": ["-map", f"0:a:{stream_id}"],
                    "stream_encoding": [
                        f"-c:a:{self.stream_count}",
                        "libfdk_aac",
                        f"-ac:a:{self.stream_count}",
                        "2",
                        f"-ar:a:{self.stream_count}",
                        str(self.target_sample_rate),
                        f"-filter:a:{self.stream_count}",
                        f'loudnorm=I={self.settings.settings["stereo_normalize"]["I"]}:'
                        f'LRA={self.settings.settings["stereo_normalize"]["LRA"]}:'
                        f'TP={self.settings.settings["stereo_normalize"]["TP"]}',
                        f"-metadata:s:a:{self.stream_count}",
                        f"language={language}",
                        f"-metadata:s:a:{self.stream_count}",
                        "title=Stereo",
                    ],
                }

                # Merge stereo args with main args
                args["stream_mapping"].extend(stereo_args["stream_mapping"])
                args["stream_encoding"].extend(stereo_args["stream_encoding"])
                self.stream_count += 1

            return args

        except Exception as e:
            logger.error(f"Error in custom_stream_mapping: {str(e)}")
            return None


def on_library_management_file_test(data):
    """Runner function - enables additional actions during the library management file tests."""
    abspath = data.get("path")
    logger.info(f"\nTesting file: {abspath}")

    try:
        probe = Probe(logger, allowed_mimetypes=["video"])
        if not probe.file(abspath):
            return data

        settings = Settings(library_id=data.get("library_id"))

        mapper = PluginStreamMapper()
        mapper.set_settings(settings)
        mapper.set_probe(probe)

        if mapper.streams_need_processing():
            data["add_file_to_pending_tasks"] = True
            logger.info("File requires audio processing")

    except Exception as e:
        logger.error(f"Error testing file: {str(e)}")
        logger.error(traceback.format_exc())

    return data


def on_worker_process(data):
    """Runner function - enables additional configured processing jobs during the worker stages of a task."""
    # Default to no FFMPEG command required
    data["exec_command"] = []
    data["repeat"] = False

    abspath = data.get("file_in")
    logger.info(f"\nProcessing file: {abspath}")

    try:
        probe = Probe(logger, allowed_mimetypes=["video"])
        if not probe.file(abspath):
            return data

        settings = Settings(library_id=data.get("library_id"))

        mapper = PluginStreamMapper()
        mapper.set_settings(settings)
        mapper.set_probe(probe)
        mapper.set_input_file(abspath)
        mapper.set_output_file(data.get("file_out"))

        if mapper.streams_need_processing():
            ffmpeg_args = mapper.get_ffmpeg_args()
            if ffmpeg_args:
                data["exec_command"] = ["ffmpeg"] + ffmpeg_args
                logger.info("FFmpeg command:\n" + " ".join(data["exec_command"]))

                parser = Parser(logger)
                parser.set_probe(probe)
                data["command_progress_parser"] = parser.parse_progress

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        logger.error(traceback.format_exc())

    return data


def on_postprocessor_task_results(data):
    """Runner function - provides a means for additional postprocessor functions based on the task success."""
    if not data.get("task_processing_success"):
        return data

    try:
        for destination_file in data.get("destination_files", []):
            logger.info(f"\nVerifying processing results for: {destination_file}")

            probe = Probe(logger, allowed_mimetypes=["video"])
            if probe.file(destination_file):
                audio_streams = [
                    s
                    for s in probe.get_probe()["streams"]
                    if s.get("codec_type") == "audio"
                ]

                # Store processing info
                info = {
                    "version": "1.0",
                    "processed_date": datetime.now().isoformat(),
                    "audio_streams": len(audio_streams),
                    "stream_details": [
                        {
                            "codec": s.get("codec_name"),
                            "channels": s.get("channels"),
                            "sample_rate": s.get("sample_rate"),
                            "language": s.get("tags", {}).get("language"),
                            "title": s.get("tags", {}).get("title"),
                        }
                        for s in audio_streams
                    ],
                }

                directory_info = UnmanicDirectoryInfo(os.path.dirname(destination_file))
                directory_info.set(
                    "seb_custom", os.path.basename(destination_file), info
                )
                directory_info.save()

    except Exception as e:
        logger.error(f"Error in post-processing: {str(e)}")
        logger.error(traceback.format_exc())

    return data
