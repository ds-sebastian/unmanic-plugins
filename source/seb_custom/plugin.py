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
    settings = {
        # General processing options
        "keep_original_streams": True,
        "force_processing": False,
        # Sample rate handling
        "target_sample_rate": 48000,  # Only convert if higher than this
        # Audio encoding - used only when re-encoding is necessary
        "stereo_encoder": "libfdk_aac",
        "multichannel_71_encoder": "opus",  # For preserving >6 channels
        "multichannel_51_encoder": "libfdk_aac",
        # Stereo downmix
        "create_stereo_tracks": True,
        "stereo_track_suffix": "AAC Stereo",
        # Loudness normalization - only applied when:
        # 1. Stream is ≤6 channels and not Atmos (can normalize without re-encoding)
        # 2. Stream needs re-encoding anyway due to sample rate
        "enable_loudnorm": True,
        "preserve_quality": True,  # New setting to control normalization behavior
        "surround_norm_i": "-24.0",
        "surround_norm_lra": "7.0",
        "surround_norm_tp": "-2.0",
        "stereo_norm_i": "-16.0",
        "stereo_norm_lra": "11.0",
        "stereo_norm_tp": "-1.0",
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

    def log_stream_info(self, analysis: AudioStreamInfo, prefix: str = "") -> None:
        """Helper to consistently log stream information"""
        logger.info(f"""{prefix}
    Stream Analysis:
        Index: {analysis.stream_index}
        Codec: {analysis.codec_name}
        Channels: {analysis.channels} ({self._get_channel_layout(analysis.channels)})
        Sample Rate: {analysis.sample_rate} Hz
        Language: {analysis.language}
        Title: {analysis.title or 'None'}
        Is Commentary: {self._is_commentary(analysis)}
        Is Atmos/TrueHD: {analysis.is_atmos}
        Bitrate: {analysis.bit_rate/1000:.1f}k per channel
    """)

    def _get_channel_layout(self, channels: int) -> str:
        """Convert channel count to human-readable layout"""
        layouts = {
            1: "Mono",
            2: "Stereo",
            6: "5.1",
            8: "7.1",
        }
        return layouts.get(channels, f"{channels} channels")

    def _is_commentary(self, analysis: AudioStreamInfo) -> bool:
        """Detect if stream is likely a commentary track"""
        if not analysis.title:
            return False
        commentary_indicators = ["commentary", "comment", "director", "cast"]
        return any(
            indicator in analysis.title.lower() for indicator in commentary_indicators
        )

    def _handle_commentary_track(self, analysis: AudioStreamInfo, stream_id: int):
        """Special handling for commentary tracks"""
        logger.info("Applying commentary track optimizations")

        # Use simpler normalization settings for commentary
        commentary_norm = "loudnorm=I=-16:LRA=11:TP=-1:measured_I=-16:measured_LRA=11:measured_TP=-1:measured_thresh=-25:offset=0.5"

        return {
            "stream_mapping": ["-map", f"0:a:{stream_id}"],
            "stream_encoding": [
                f"-c:a:{stream_id}",
                "copy",
                f"-filter:a:{stream_id}",
                commentary_norm,
            ],
        }

    def _handle_atmos_track(
        self,
        analysis: AudioStreamInfo,
        stream_id: int,
        needs_sample_rate_conversion: bool,
    ):
        """Special handling for Atmos/TrueHD content"""
        if needs_sample_rate_conversion:
            logger.warning(
                "Atmos content needs sample rate conversion - will lose Atmos metadata"
            )
            # Convert to high-quality multichannel
            return self._create_multichannel_mapping(analysis, stream_id)
        else:
            logger.info("Preserving Atmos content (no conversion needed)")
            return {
                "stream_mapping": ["-map", f"0:a:{stream_id}"],
                "stream_encoding": [f"-c:a:{stream_id}", "copy"],
            }

    def _create_stereo_downmix(self, analysis: AudioStreamInfo, stream_id: int):
        """Create stereo downmix mapping"""
        next_stream_id = len(self.streams_by_language)
        stereo_encoder = self.settings.get_setting("stereo_encoder")
        stereo_bitrate = self.calculate_bitrate(analysis, target_channels=2)

        logger.info(f"""
    Creating stereo downmix:
        Original: {analysis.channels} channels
        Encoder: {stereo_encoder}
        Bitrate: {stereo_bitrate}k
        Stream ID: {next_stream_id}
    """)

        return {
            "stream_mapping": ["-map", f"0:a:{stream_id}"],
            "stream_encoding": [
                f"-c:a:{next_stream_id}",
                stereo_encoder,
                f"-ac:a:{next_stream_id}",
                "2",
                f"-ar:a:{next_stream_id}",
                str(self.settings.get_setting("target_sample_rate")),
                f"-b:a:{next_stream_id}",
                f"{stereo_bitrate}k",
                f"-filter:a:{next_stream_id}",
                self.get_loudnorm_filter(is_stereo=True),
                f"-metadata:s:a:{next_stream_id}",
                f'title={f"{analysis.title} " if analysis.title else ""}{self.settings.get_setting("stereo_track_suffix")}',
            ],
        }

    def _create_multichannel_mapping(self, analysis: AudioStreamInfo, stream_id: int):
        """Convert multichannel to appropriate format"""
        encoder = self.get_encoder_for_stream(analysis)
        bitrate = self.calculate_bitrate(analysis)

        logger.info(f"""
    Converting multichannel stream:
        Channels: {analysis.channels}
        Encoder: {encoder}
        Bitrate: {bitrate}k
        Sample Rate: {self.settings.get_setting('target_sample_rate')}
    """)

        return {
            "stream_mapping": ["-map", f"0:a:{stream_id}"],
            "stream_encoding": [
                f"-c:a:{stream_id}",
                encoder,
                f"-ar:a:{stream_id}",
                str(self.settings.get_setting("target_sample_rate")),
                f"-b:a:{stream_id}",
                f"{bitrate}k",
                # Apply normalization if possible and enabled
                *(
                    [
                        f"-filter:a:{stream_id}",
                        self.get_loudnorm_filter(
                            is_stereo=False, channels=analysis.channels
                        ),
                    ]
                    if self.settings.get_setting("enable_loudnorm")
                    and analysis.channels <= 6
                    else []
                ),
            ],
        }

    def set_settings(self, settings: Settings) -> None:
        """Set plugin settings"""
        self.settings = settings

    def analyze_streams(self, probe_streams: List[dict]) -> None:
        """Analyze all audio streams and categorize them"""
        logger.info("Beginning comprehensive audio stream analysis...")

        # Track statistics for logging
        stats = {
            "total_streams": 0,
            "high_samplerate_streams": 0,
            "multichannel_streams": 0,
            "commentary_tracks": 0,
            "atmos_tracks": 0,
        }

        for stream in probe_streams:
            if stream.get("codec_type") != "audio":
                continue

            stats["total_streams"] += 1
            stream_info = AudioStreamInfo(stream)
            self.log_stream_info(
                stream_info, prefix=f"\nAnalyzing Stream {stream_info.stream_index}"
            )

            # Collect statistics
            if stream_info.sample_rate > 48000:
                stats["high_samplerate_streams"] += 1
            if stream_info.channels > 2:
                stats["multichannel_streams"] += 1
            if self._is_commentary(stream_info):
                stats["commentary_tracks"] += 1
            if stream_info.is_atmos:
                stats["atmos_tracks"] += 1

            # Track streams by language
            if stream_info.language not in self.streams_by_language:
                self.streams_by_language[stream_info.language] = []
                logger.debug(f"First stream found for language: {stream_info.language}")
            self.streams_by_language[stream_info.language].append(stream_info)

            # Track existing stereo streams
            if stream_info.channels == 2:
                if stream_info.language in self.stereo_streams_by_language:
                    logger.warning(
                        f"Multiple stereo streams found for language: {stream_info.language}"
                    )
                self.stereo_streams_by_language[stream_info.language] = stream_info

        # Log analysis summary
        logger.info(f"""
    Stream Analysis Summary:
        Total Audio Streams: {stats['total_streams']}
        Languages Found: {list(self.streams_by_language.keys())}
        High Sample Rate (>48kHz) Streams: {stats['high_samplerate_streams']}
        Multichannel Streams: {stats['multichannel_streams']}
        Commentary Tracks: {stats['commentary_tracks']}
        Atmos/TrueHD Tracks: {stats['atmos_tracks']}
        Existing Stereo Streams: {len(self.stereo_streams_by_language)}
    """)

    def test_stream_needs_processing(self, stream_info: dict) -> bool:
        """Determine if stream needs processing"""
        analysis = AudioStreamInfo(stream_info)

        logger.debug(f"""
    Processing decision for stream:
        Index: {analysis.stream_index}
        Codec: {analysis.codec_name}
        Channels: {analysis.channels}
        Sample Rate: {analysis.sample_rate}
        Is Atmos/TrueHD: {analysis.is_atmos}
        Language: {analysis.language}
    """)

        # Critical compatibility check first - always convert if sample rate too high
        if analysis.sample_rate > self.settings.get_setting("target_sample_rate"):
            logger.info(
                f"Must process: Sample rate {analysis.sample_rate} exceeds device maximum {self.settings.get_setting('target_sample_rate')}Hz"
            )
            return True

        # Force processing override
        if self.settings.get_setting("force_processing"):
            logger.info("Processing: Force processing enabled")
            return True

        # Normalize only if we can do it without re-encoding and quality preservation is off
        if (
            self.settings.get_setting("enable_loudnorm")
            and not self.settings.get_setting("preserve_quality")
            and analysis.channels <= 6
            and not analysis.is_atmos
        ):
            logger.info("Processing: Applying normalization to compatible stream")
            return True

        # Create stereo downmix if needed
        if (
            analysis.channels > 2
            and self.settings.get_setting("create_stereo_tracks")
            and analysis.language not in self.stereo_streams_by_language
        ):
            logger.info(
                f"Processing: Need stereo version for language {analysis.language}"
            )
            return True

        # Log why we're preserving the stream as-is
        if analysis.is_atmos:
            logger.info("Preserving Atmos/TrueHD stream without modification")
        elif analysis.channels > 6:
            logger.info(f"Preserving {analysis.channels} channel stream (>6 channels)")
        elif self.settings.get_setting("preserve_quality"):
            logger.info("Preserving original quality (normalization skipped)")

        return False

    def custom_stream_mapping(self, stream_info: dict, stream_id: int):
        """Bridge between StreamMapper and our build_stream_mapping"""
        logger.debug(f"Generating mapping for stream {stream_id}")
        mapping = self.build_stream_mapping(stream_info, stream_id)

        # Log the mapping that will be used
        logger.debug(f"""
    Stream mapping for stream {stream_id}:
        Mapping args: {' '.join(mapping['stream_mapping'])}
        Encoding args: {' '.join(mapping['stream_encoding'])}
    """)

        return mapping

    def get_ffmpeg_args(self) -> List[str]:
        """Override parent's get_ffmpeg_args to ensure proper stream order"""
        # Start with input args
        output_args = []

        # Process all streams in order
        if self.streams_to_map:
            logger.debug(f"Processing {len(self.streams_to_map)} streams")
            for stream_id in self.streams_to_map:
                stream_info = self.probe.get_stream_info(stream_id)
                mapping = self.custom_stream_mapping(stream_info, stream_id)

                if mapping.get("stream_mapping"):
                    output_args.extend(mapping["stream_mapping"])
                if mapping.get("stream_encoding"):
                    output_args.extend(mapping["stream_encoding"])

        # Add all default args
        args = []
        args.extend(self.main_options)
        args.extend(self.input_file)
        args.extend(self.advanced_options)
        args.extend(output_args)
        args.extend(["-y", self.output_file])

        return args

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

    def get_loudnorm_filter(self, is_stereo: bool = False, channels: int = 2) -> str:
        """Get loudnorm filter string based on channel count"""
        if not self.settings.get_setting("enable_loudnorm"):
            return None

        # Check if we can apply normalization
        if channels > 6:
            logger.warning(
                f"Cannot apply normalization to {channels} channels without re-encoding"
            )
            return None

        if is_stereo:
            filter_string = (
                f"loudnorm=I={self.settings.get_setting('stereo_norm_i')}:"
                f"LRA={self.settings.get_setting('stereo_norm_lra')}:"
                f"TP={self.settings.get_setting('stereo_norm_tp')}"
            )
            logger.debug(f"Using stereo normalization: {filter_string}")
            return filter_string
        else:
            filter_string = (
                f"loudnorm=I={self.settings.get_setting('surround_norm_i')}:"
                f"LRA={self.settings.get_setting('surround_norm_lra')}:"
                f"TP={self.settings.get_setting('surround_norm_tp')}"
            )
            logger.debug(f"Using surround normalization: {filter_string}")
            return filter_string

    def build_stream_mapping(self, stream_info: dict, stream_id: int):
        """Build FFmpeg mapping for stream processing"""
        analysis = AudioStreamInfo(stream_info)

        stream_mapping = []
        stream_encoding = []

        # Map original stream
        stream_mapping.extend(["-map", f"0:a:{stream_id}"])

        needs_sample_rate_conversion = analysis.sample_rate > self.settings.get_setting(
            "target_sample_rate"
        )
        can_normalize_without_reencoding = (
            analysis.channels <= 6
            and not analysis.is_atmos
            and not self.settings.get_setting("preserve_quality")
        )

        logger.debug(f"""
    Stream processing plan:
        Sample rate conversion needed: {needs_sample_rate_conversion}
        Can normalize without re-encoding: {can_normalize_without_reencoding}
        Is Atmos/TrueHD: {analysis.is_atmos}
        Channels: {analysis.channels}
    """)

        if needs_sample_rate_conversion:
            # Must re-encode - use best quality options
            if analysis.channels > 6:
                encoder = self.settings.get_setting("multichannel_71_encoder")  # OPUS
                logger.info(
                    f"Converting to {encoder} to preserve {analysis.channels} channels"
                )
            else:
                encoder = self.settings.get_setting(
                    "multichannel_51_encoder"
                )  # libfdk_aac
                logger.info(f"Converting to {encoder} for best quality")

            stream_encoding.extend(
                [
                    f"-c:a:{stream_id}",
                    encoder,
                    f"-ar:a:{stream_id}",
                    str(self.settings.get_setting("target_sample_rate")),
                ]
            )

            # Apply normalization since we're re-encoding anyway
            if self.settings.get_setting("enable_loudnorm"):
                loudnorm = self.get_loudnorm_filter(
                    is_stereo=analysis.channels <= 2, channels=analysis.channels
                )
                if loudnorm:
                    stream_encoding.extend([f"-filter:a:{stream_id}", loudnorm])
        else:
            # Can we normalize without re-encoding?
            if can_normalize_without_reencoding:
                stream_encoding.extend(
                    [
                        f"-c:a:{stream_id}",
                        "copy",
                        f"-filter:a:{stream_id}",
                        self.get_loudnorm_filter(is_stereo=analysis.channels <= 2),
                    ]
                )
                logger.info("Applying normalization to copied stream")
            else:
                # Preserve original quality
                stream_encoding.extend([f"-c:a:{stream_id}", "copy"])
                logger.info("Preserving original stream quality")

        # Handle stereo downmix if needed
        if (
            analysis.channels > 2
            and self.settings.get_setting("create_stereo_tracks")
            and analysis.language not in self.stereo_streams_by_language
        ):
            logger.info("Creating additional stereo version for compatibility")
            stereo_mapping = self._create_stereo_downmix(analysis, stream_id)
            stream_mapping.extend(stereo_mapping["stream_mapping"])
            stream_encoding.extend(stereo_mapping["stream_encoding"])

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
