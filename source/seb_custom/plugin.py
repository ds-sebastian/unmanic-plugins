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
- Applies normalization only during re-encoding
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
        self.stereo_streams_by_language = {}  # Track existing stereo streams
        self.stream_count = 0  # Track output stream count
        self.settings = None
        logger.info("Initialized PluginStreamMapper")

    def set_settings(self, settings):
        """Configure settings object"""
        self.settings = settings

    def test_stream_needs_processing(self, stream_info: dict) -> bool:
        try:
            # First check if it's an audio stream
            if stream_info.get("codec_type") != "audio":
                return False

            # Convert values with error handling
            try:
                sample_rate = int(str(stream_info.get("sample_rate", "48000")))
                channels = int(str(stream_info.get("channels", "2")))
            except (ValueError, TypeError) as e:
                logger.error(f"Error converting stream values: {e}")
                return False

            logger.debug(f"""Testing stream:
        Sample Rate: {sample_rate}
        Channels: {channels}
        Target Rate: {self.target_sample_rate}
        Codec: {stream_info.get('codec_name')}""")

            # Always process if sample rate is too high
            if sample_rate > self.target_sample_rate:
                logger.debug(
                    f"Stream needs processing - high sample rate: {sample_rate}"
                )
                return True

            # Process multichannel if we don't have a stereo version
            if channels > 2:
                # Check if we already have a stereo version for this language
                stream_language = stream_info.get("tags", {}).get("language", "und")
                if stream_language not in self.stereo_streams_by_language:
                    logger.debug("Stream needs processing - needs stereo version")
                    return True

            return False

        except Exception as e:
            logger.error(f"Error testing stream: {str(e)}")
            logger.error(traceback.format_exc())
            return False

    def analyze_streams(self, probe_streams: list) -> dict:
        """Analyze all audio streams and return detailed info"""
        audio_streams = []
        stereo_streams_by_language = {}

        logger.info(f"\nAnalyzing {len(probe_streams)} streams:")

        for stream in probe_streams:
            if stream.get("codec_type") != "audio":
                continue

            try:
                audio_index = stream.get("index", 0)
                logger.info(f"\nAnalyzing audio stream {audio_index}:")

                # Fix: Properly indent stream_info within try block
                stream_info = {
                    "index": audio_index,
                    "absolute_index": len(audio_streams),
                    "codec": stream.get("codec_name", "").lower(),
                    "channels": int(stream.get("channels", 2)),
                    "sample_rate": int(stream.get("sample_rate", "48000")),
                    "bit_rate": stream.get("bit_rate"),
                    "language": stream.get("tags", {}).get("language", "und"),
                    "title": stream.get("tags", {}).get("title", ""),
                    "disposition": stream.get("disposition", {}),
                    "tags": stream.get("tags", {}),
                }

                # Log stream details
                logger.info(f"""Stream details:
        Index: {stream_info['index']} (Absolute: {stream_info['absolute_index']})
        Codec: {stream_info['codec']}
        Channels: {stream_info['channels']}
        Sample Rate: {stream_info['sample_rate']} Hz
        Bit Rate: {stream_info['bit_rate'] if stream_info['bit_rate'] else 'Unknown'}
        Language: {stream_info['language']}
        Title: {stream_info['title'] or 'None'}
        Tags: {stream_info['tags']}
    """)

                # Track stereo streams by language
                if stream_info["channels"] == 2:
                    if stream_info["language"] not in stereo_streams_by_language:
                        stereo_streams_by_language[stream_info["language"]] = (
                            stream_info
                        )
                        logger.debug(
                            f"Found stereo stream for language {stream_info['language']}"
                        )

                audio_streams.append(stream_info)

            except Exception as e:
                logger.error(f"Error analyzing stream {audio_index}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

        return {
            "audio_streams": audio_streams,
            "stereo_streams_by_language": stereo_streams_by_language,
        }

    def determine_stream_processing(
        self, stream_info: dict, is_stereo_exists: bool
    ) -> dict:
        """
        Determine how each stream should be processed
        Returns dict with processing instructions
        """
        try:
            needs_processing = False
            process_type = "copy"
            encoder = None
            reasons = []
            bitrate = None
            stereo_bitrate = None

            logger.info(f"""
    Determining processing for stream:
        Index: {stream_info['index']}
        Channels: {stream_info['channels']}
        Sample Rate: {stream_info['sample_rate']} Hz
        Language: {stream_info['language']}
    """)

            # Check if sample rate conversion needed
            try:
                needs_sample_rate_conversion = (
                    int(stream_info["sample_rate"]) > self.target_sample_rate
                )
            except (ValueError, TypeError):
                logger.error(f"Invalid sample rate value: {stream_info['sample_rate']}")
                needs_sample_rate_conversion = False

            if needs_sample_rate_conversion:
                needs_processing = True
                process_type = "convert"
                reasons.append(
                    f"Sample rate {stream_info['sample_rate']}Hz > {self.target_sample_rate}Hz target"
                )

                # Choose encoder based on channels
                if stream_info["channels"] > 6:
                    encoder = "libopus"
                    logger.info("Using OPUS encoder for >6 channels")
                else:
                    encoder = "libfdk_aac"
                    logger.info("Using libfdk_aac encoder for conversion")

            # Determine if we need stereo version
            needs_stereo = False
            if stream_info["channels"] > 2 and not is_stereo_exists:
                needs_stereo = True
                reasons.append(
                    f"Creating stereo version for {stream_info['channels']} channels"
                )
                logger.info(
                    f"Will create stereo version for language {stream_info['language']}"
                )

            # Calculate bitrates if needed
            if needs_processing or needs_stereo:
                if stream_info.get("bit_rate"):
                    try:
                        # Handle both string and integer bit_rate values
                        source_bitrate = str(stream_info["bit_rate"])
                        if source_bitrate.endswith("k"):
                            source_bitrate = int(source_bitrate.rstrip("k")) * 1000
                        else:
                            source_bitrate = int(source_bitrate)

                        per_channel_bitrate = int(
                            source_bitrate / (1000 * stream_info["channels"])
                        )

                        if needs_processing:
                            bitrate = (
                                f"{per_channel_bitrate * stream_info['channels']}k"
                            )
                        if needs_stereo:
                            stereo_bitrate = f"{per_channel_bitrate * 2}k"

                        logger.info(f"""Bitrate calculation:
        Source: {source_bitrate/1000:.1f}k
        Per channel: {per_channel_bitrate}k
        Main bitrate: {bitrate}
        Stereo bitrate: {stereo_bitrate}
    """)
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error calculating bitrate: {str(e)}")
                        logger.debug("Will use FFmpeg defaults")
                        bitrate = None
                        stereo_bitrate = None
                else:
                    logger.info("No source bitrate available, will use FFmpeg defaults")

            return {
                "needs_processing": needs_processing,
                "process_type": process_type,
                "encoder": encoder,
                "reasons": reasons,
                "needs_stereo": needs_stereo,
                "bitrate": bitrate,
                "stereo_bitrate": stereo_bitrate,
            }

        except Exception as e:
            logger.error(f"Error determining stream processing: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "needs_processing": False,
                "process_type": "copy",
                "encoder": None,
                "reasons": ["Error in processing determination"],
                "needs_stereo": False,
                "bitrate": None,
                "stereo_bitrate": None,
            }

    def build_stream_mappings(
        self, audio_streams: list, stereo_streams_by_language: dict
    ) -> list:
        """
        Build complete FFmpeg mappings for all streams
        Returns list of mapping arguments
        """
        try:
            # Base FFmpeg args
            ffmpeg_args = [
                "-hide_banner",
                "-loglevel",
                "info",
                "-i",
                self.input_file,
                "-map",
                "0:v",
                "-c:v",
                "copy",  # Copy video
                "-map",
                "0:s?",
                "-c:s",
                "copy",  # Copy subtitles
                "-map_chapters",
                "0",  # Keep chapters
            ]

            # Reset stream counter
            self.stream_count = 0

            # Process each audio stream
            for stream in audio_streams:
                try:
                    # Check if we already have stereo for this language
                    has_stereo = stream["language"] in stereo_streams_by_language

                    # Determine processing needs
                    processing = self.determine_stream_processing(stream, has_stereo)

                    # Map original stream
                    ffmpeg_args.extend(["-map", f'0:a:{stream["absolute_index"]}'])

                    if processing["needs_processing"]:
                        # Convert stream
                        stream_args = [
                            f"-c:a:{self.stream_count}",
                            processing["encoder"],
                            f"-ar:a:{self.stream_count}",
                            str(self.target_sample_rate),
                            f"-filter:a:{self.stream_count}",
                            f"loudnorm=I={self.settings.settings['surround_normalize']['I']}:"
                            f"LRA={self.settings.settings['surround_normalize']['LRA']}:"
                            f"TP={self.settings.settings['surround_normalize']['TP']}",
                        ]

                        # Add bitrate if calculated
                        if processing["bitrate"]:
                            stream_args.extend(
                                [f"-b:a:{self.stream_count}", processing["bitrate"]]
                            )

                        ffmpeg_args.extend(stream_args)
                    else:
                        # Just copy the stream
                        ffmpeg_args.extend([f"-c:a:{self.stream_count}", "copy"])

                    # Add metadata
                    if stream["language"] != "und":
                        ffmpeg_args.extend(
                            [
                                f"-metadata:s:a:{self.stream_count}",
                                f'language={stream["language"]}',
                            ]
                        )
                    if stream.get("title"):
                        ffmpeg_args.extend(
                            [
                                f"-metadata:s:a:{self.stream_count}",
                                f'title={stream["title"]}',
                            ]
                        )

                    self.stream_count += 1

                except Exception as e:
                    logger.error(f"Error processing stream {stream['index']}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue

            # Add final output arguments
            ffmpeg_args.extend(
                ["-max_muxing_queue_size", "4096", "-y", self.output_file]
            )

            return ffmpeg_args

        except Exception as e:
            logger.error(f"Error building stream mappings: {str(e)}")
            logger.error(traceback.format_exc())
            return []


def on_library_management_file_test(data):
    """
    Runner function - enables additional actions during the library management file tests.
    """
    abspath = data.get("path")
    logger.info(f"\nTesting file: {abspath}")

    try:
        probe = Probe(logger, allowed_mimetypes=["video"])
        if not probe.file(abspath):
            logger.error("Failed to probe file")
            return data

        # Get stream mapper
        mapper = PluginStreamMapper()
        mapper.set_probe(probe)

        # Check audio streams
        audio_streams = [
            s for s in probe.get_probe()["streams"] if s.get("codec_type") == "audio"
        ]

        if not audio_streams:
            logger.info("No audio streams found")
            return data

        logger.info(f"Found {len(audio_streams)} audio streams")

        # Check if any stream needs processing
        needs_processing = False
        for stream in audio_streams:
            if mapper.test_stream_needs_processing(stream):
                needs_processing = True
                break

        if needs_processing:
            data["add_file_to_pending_tasks"] = True
            logger.info("File added to task list")
        else:
            logger.info("No processing needed")

    except Exception as e:
        logger.error(f"Error testing file: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

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
            logger.error("Failed to probe file")
            return data

        # Configure settings object
        settings = Settings(library_id=data.get("library_id"))

        # Initialize mapper with settings and probe
        mapper = PluginStreamMapper()
        mapper.set_settings(settings)
        mapper.set_probe(probe)

        # Check if any streams need processing
        audio_streams = [
            s for s in probe.get_probe()["streams"] if s.get("codec_type") == "audio"
        ]

        if not any(mapper.test_stream_needs_processing(s) for s in audio_streams):
            logger.info("No streams need processing")
            return data

        # Build ffmpeg command
        mapper.set_input_file(abspath)
        mapper.set_output_file(data.get("file_out"))

        ffmpeg_args = [
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            abspath,
            "-map",
            "0:v",
            "-c:v",
            "copy",  # Copy video
            "-map",
            "0:s?",
            "-c:s",
            "copy",  # Copy subtitles
            "-map_chapters",
            "0",  # Keep chapters
        ]

        # Process all audio streams
        streams_info = mapper.analyze_streams(probe.get_probe()["streams"])
        stream_mappings = mapper.build_stream_mappings(
            streams_info["audio_streams"], streams_info["stereo_streams_by_language"]
        )
        ffmpeg_args.extend(stream_mappings)
        # Add output file
        ffmpeg_args.extend(
            ["-max_muxing_queue_size", "4096", "-y", data.get("file_out")]
        )

        # Log full command
        logger.info("FFmpeg command:")
        logger.info(" ".join(["ffmpeg"] + ffmpeg_args))

        # Set the execution command
        data["exec_command"] = ["ffmpeg"]
        data["exec_command"].extend(ffmpeg_args)

        # Set up the parser
        parser = Parser(logger)
        parser.set_probe(probe)
        data["command_progress_parser"] = parser.parse_progress

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

    return data


def on_postprocessor_task_results(data):
    """
    Runner function - provides a means for additional postprocessor functions based on the task success.
    """
    if not data.get("task_processing_success"):
        logger.debug("Task processing was not successful")
        return data

    try:
        for destination_file in data.get("destination_files", []):
            logger.info(f"\nVerifying processing results for: {destination_file}")

            directory_info = UnmanicDirectoryInfo(os.path.dirname(destination_file))

            # Verify results
            probe = Probe(logger, allowed_mimetypes=["video"])
            if probe.file(destination_file):
                audio_streams = [
                    s
                    for s in probe.get_probe()["streams"]
                    if s.get("codec_type") == "audio"
                ]

                # Log detailed stream information
                logger.info(
                    f"Final audio configuration ({len(audio_streams)} streams):"
                )
                for idx, stream in enumerate(audio_streams):
                    logger.info(f"""Stream {idx}:
    Codec: {stream.get('codec_name')}
    Channels: {stream.get('channels')}
    Sample Rate: {stream.get('sample_rate')}
    Bit Rate: {stream.get('bit_rate', 'N/A')}
    Language: {stream.get('tags', {}).get('language', 'und')}
    Title: {stream.get('tags', {}).get('title', 'None')}
""")

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
                            "bit_rate": s.get("bit_rate"),
                            "language": s.get("tags", {}).get("language"),
                            "title": s.get("tags", {}).get("title"),
                            "disposition": s.get("disposition", {}),
                            "tags": s.get("tags", {}),
                        }
                        for s in audio_streams
                    ],
                }

                directory_info.set(
                    "seb_custom", os.path.basename(destination_file), info
                )
                directory_info.save()
                logger.info("Saved processing info to directory info")

                # Verify expected results
                has_high_sample_rate = any(
                    int(str(s.get("sample_rate", "0"))) > 48000 for s in audio_streams
                )
                has_multichannel = any(
                    int(str(s.get("channels", "2"))) > 2 for s in audio_streams
                )

                if has_high_sample_rate:
                    logger.error(
                        "Found streams with sample rate > 48kHz after processing!"
                    )
                    for s in audio_streams:
                        try:
                            sample_rate = int(str(s.get("sample_rate", "0")))
                            if sample_rate > 48000:
                                logger.error(f"Stream still has high sample rate: {s}")
                        except (ValueError, TypeError) as e:
                            logger.error(
                                f"Error converting sample rate '{s.get('sample_rate')}': {e}"
                            )

                if has_multichannel and not any(
                    int(str(s.get("channels", "2"))) == 2 for s in audio_streams
                ):
                    logger.error(
                        "Multichannel content present but no stereo version found!"
                    )

    except Exception as e:
        logger.error(f"Error in post-processing: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

    return data
