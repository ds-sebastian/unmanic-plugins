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
    settings = {}  # No configurable settings - everything is hardcoded


class PluginStreamMapper(StreamMapper):
    def __init__(self):
        super(PluginStreamMapper, self).__init__(logger, ["audio"])
        self.target_sample_rate = 48000
        self.stereo_streams_by_language = {}  # Track existing stereo streams
        self.stream_count = 0  # Track output stream count
        logger.info("Initialized PluginStreamMapper")

    def analyze_stream(self, stream_info: dict) -> dict:
        """Analyze single stream and return relevant info"""
        try:
            stream_id = stream_info.get("index", 0)
            logger.info(f"\nAnalyzing stream {stream_id}:")

            # Extract all possible useful information
            analysis = {
                "index": stream_id,
                "codec": stream_info.get("codec_name", "").lower(),
                "codec_long_name": stream_info.get("codec_long_name", ""),
                "channels": int(stream_info.get("channels", 2)),
                "sample_rate": int(stream_info.get("sample_rate", 48000)),
                "bit_rate": stream_info.get("bit_rate"),
                "tags": stream_info.get("tags", {}),
                "language": stream_info.get("tags", {}).get("language", "und"),
                "title": stream_info.get("tags", {}).get("title", ""),
                "is_atmos": "atmos" in str(stream_info).lower(),
            }

            logger.info(f"""Stream details:
    Codec: {analysis['codec']} ({analysis['codec_long_name']})
    Channels: {analysis['channels']}
    Sample Rate: {analysis['sample_rate']} Hz
    Bit Rate: {analysis['bit_rate'] if analysis['bit_rate'] else 'Unknown'}
    Language: {analysis['language']}
    Title: {analysis['title'] or 'None'}
    Atmos: {analysis['is_atmos']}
    All Tags: {analysis['tags']}
""")
            return analysis

        except Exception as e:
            logger.error(
                f"Error analyzing stream {stream_info.get('index', 'Unknown')}: {str(e)}"
            )
            logger.error(f"Stream info: {stream_info}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def test_stream_needs_processing(self, stream_info: dict) -> bool:
        """Determine if stream needs processing"""
        analysis = self.analyze_stream(stream_info)
        if not analysis:
            return False

        needs_processing = False
        reasons = []

        # Check sample rate
        if analysis["sample_rate"] > self.target_sample_rate:
            needs_processing = True
            reasons.append(
                f"Sample rate {analysis['sample_rate']}Hz > {self.target_sample_rate}Hz target"
            )

        # Check if we need a stereo version
        if (
            analysis["channels"] > 2
            and analysis["language"] not in self.stereo_streams_by_language
        ):
            needs_processing = True
            reasons.append(
                f"Need stereo version for {analysis['channels']} channel stream"
            )

        if needs_processing:
            logger.info(
                f"Stream {analysis['index']} needs processing: {', '.join(reasons)}"
            )
        else:
            logger.info(f"Stream {analysis['index']} does not need processing")

        return needs_processing

    def calculate_stream_bitrate(self, analysis: dict) -> tuple:
        """Calculate bitrates and return (per_channel_bitrate, total_bitrate)"""
        try:
            if not analysis["bit_rate"]:
                logger.info("No source bitrate available, will use FFmpeg defaults")
                return None, None

            source_bitrate = int(analysis["bit_rate"])
            per_channel = int(source_bitrate / (1000 * analysis["channels"]))

            logger.info(f"""Bitrate calculation:
    Source total: {source_bitrate/1000:.1f}k
    Per channel: {per_channel}k
    Channels: {analysis['channels']}
""")

            return per_channel, f"{per_channel * analysis['channels']}k"

        except Exception as e:
            logger.error(f"Error calculating bitrate: {str(e)}")
            logger.error(f"Analysis data: {analysis}")
            return None, None

    def custom_stream_mapping(self, stream_info: dict, stream_id: int) -> dict:
        """Build stream mapping command"""
        try:
            analysis = self.analyze_stream(stream_info)
            if not analysis:
                logger.warning(f"Stream analysis failed, copying stream {stream_id}")
                return {
                    "stream_mapping": ["-map", f"0:a:{stream_id}"],
                    "stream_encoding": [f"-c:a:{self.stream_count}", "copy"],
                }

            stream_mapping = []
            stream_encoding = []

            # MAIN STREAM HANDLING
            stream_mapping.extend(["-map", f"0:a:{stream_id}"])

            if analysis["sample_rate"] > self.target_sample_rate:
                # Need to convert sample rate - calculate bitrate first
                per_channel_bitrate, total_bitrate = self.calculate_stream_bitrate(
                    analysis
                )

                # Select encoder based on channel count
                if analysis["channels"] > 6:
                    encoder = "libopus"
                    logger.info(f"""Converting stream with OPUS:
    Original: {analysis['sample_rate']}Hz {analysis['channels']} channels
    Target: {self.target_sample_rate}Hz (OPUS will handle bitrate)
    Applying normalization
""")
                    stream_encoding.extend(
                        [
                            f"-c:a:{self.stream_count}",
                            encoder,
                            f"-ar:a:{self.stream_count}",
                            str(self.target_sample_rate),
                            f"-filter:a:{self.stream_count}",
                            "loudnorm=I=-24:LRA=7:TP=-2",
                        ]
                    )
                else:
                    encoder = "libfdk_aac"
                    logger.info(f"""Converting stream with libfdk_aac:
    Original: {analysis['sample_rate']}Hz {analysis['channels']} channels
    Target: {self.target_sample_rate}Hz
    Bitrate: {total_bitrate if total_bitrate else 'FFmpeg default'}
    Applying normalization
""")
                    stream_encoding.extend(
                        [
                            f"-c:a:{self.stream_count}",
                            encoder,
                            f"-ar:a:{self.stream_count}",
                            str(self.target_sample_rate),
                            f"-filter:a:{self.stream_count}",
                            "loudnorm=I=-24:LRA=7:TP=-2",
                        ]
                    )
                    if total_bitrate:
                        stream_encoding.extend(
                            [f"-b:a:{self.stream_count}", total_bitrate]
                        )
            else:
                # If sample rate is fine, just copy the stream
                logger.info(f"""Keeping original stream:
    Sample rate: {analysis['sample_rate']}Hz
    Channels: {analysis['channels']}
    No processing needed
""")
                stream_encoding.extend([f"-c:a:{self.stream_count}", "copy"])

            # Preserve metadata for main stream
            if analysis["language"] != "und":
                stream_encoding.extend(
                    [
                        f"-metadata:s:a:{self.stream_count}",
                        f'language={analysis["language"]}',
                    ]
                )
            if analysis["title"]:
                stream_encoding.extend(
                    [f"-metadata:s:a:{self.stream_count}", f'title={analysis["title"]}']
                )

            self.stream_count += 1

            # STEREO VERSION (only if original is multichannel)
            if (
                analysis["channels"] > 2
                and analysis["language"] not in self.stereo_streams_by_language
            ):
                # Calculate stereo bitrate
                per_channel_bitrate, _ = self.calculate_stream_bitrate(analysis)
                stereo_bitrate = (
                    f"{per_channel_bitrate * 2}k" if per_channel_bitrate else None
                )

                logger.info(f"""Creating stereo version:
    Source: {analysis['channels']} channels
    Language: {analysis['language']}
    Bitrate: {stereo_bitrate if stereo_bitrate else 'FFmpeg default'}
    Applying normalization
""")

                # Add stereo stream mapping
                stream_mapping.extend(["-map", f"0:a:{stream_id}"])

                stereo_encoding = [
                    f"-c:a:{self.stream_count}",
                    "libfdk_aac",
                    f"-ac:a:{self.stream_count}",
                    "2",
                    f"-ar:a:{self.stream_count}",
                    str(self.target_sample_rate),
                    f"-filter:a:{self.stream_count}",
                    "loudnorm=I=-16:LRA=11:TP=-1",
                ]

                if stereo_bitrate:
                    stereo_encoding.extend(
                        [f"-b:a:{self.stream_count}", stereo_bitrate]
                    )

                stream_encoding.extend(stereo_encoding)

                # Add metadata to stereo stream
                stream_encoding.extend(
                    [f"-metadata:s:a:{self.stream_count}", "title=Stereo"]
                )
                if analysis["language"] != "und":
                    stream_encoding.extend(
                        [
                            f"-metadata:s:a:{self.stream_count}",
                            f'language={analysis["language"]}',
                        ]
                    )

                self.stereo_streams_by_language[analysis["language"]] = True
                self.stream_count += 1

            return {
                "stream_mapping": stream_mapping,
                "stream_encoding": stream_encoding,
            }

        except Exception as e:
            logger.error(f"Error in stream mapping: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "stream_mapping": ["-map", f"0:a:{stream_id}"],
                "stream_encoding": [f"-c:a:{self.stream_count}", "copy"],
            }


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
    """
    Runner function - enables additional configured processing jobs during the worker stages of a task.
    """
    abspath = data.get("file_in")
    logger.info(f"\nProcessing file: {abspath}")

    try:
        probe = Probe(logger, allowed_mimetypes=["video"])
        if not probe.file(abspath):
            logger.error("Failed to probe file")
            return data

        mapper = PluginStreamMapper()
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
        for stream in audio_streams:
            stream_id = stream.get("index")
            mapping = mapper.custom_stream_mapping(stream, stream_id)
            if mapping:
                ffmpeg_args.extend(mapping["stream_mapping"])
                ffmpeg_args.extend(mapping["stream_encoding"])

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
                    s.get("sample_rate", 0) > 48000 for s in audio_streams
                )
                has_multichannel = any(s.get("channels", 0) > 2 for s in audio_streams)

                if has_high_sample_rate:
                    logger.error(
                        "Found streams with sample rate > 48kHz after processing!"
                    )
                    for s in audio_streams:
                        if s.get("sample_rate", 0) > 48000:
                            logger.error(f"Stream still has high sample rate: {s}")

                if has_multichannel and not any(
                    s.get("channels") == 2 for s in audio_streams
                ):
                    logger.error(
                        "Multichannel content present but no stereo version found!"
                    )

            else:
                logger.error(f"Failed to probe processed file: {destination_file}")

    except Exception as e:
        logger.error(f"Error in post-processing: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

    return data
