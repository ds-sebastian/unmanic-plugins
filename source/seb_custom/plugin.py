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
from typing import List

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
        """Improved Atmos detection"""
        codec_name = stream_info.get("codec_name", "").lower()
        format_name = stream_info.get("format_name", "").lower()
        profile = stream_info.get("profile", "").lower()
        tags = stream_info.get("tags", {})

        return any(
            [
                "truehd" in codec_name and "atmos" in format_name,
                "eac3" in codec_name and "atmos" in format_name,
                "atmos" in tags.get("format", "").lower(),
                "atmos" in profile,
            ]
        )


class PluginStreamMapper(StreamMapper):
    """Handles stream analysis and processing decisions"""

    def __init__(self):
        super(PluginStreamMapper, self).__init__(logger, ["audio"])
        self.settings = None
        self.streams_to_map = []  # Initialize the missing attribute
        self.streams_by_language = {}
        self.stereo_streams_by_language = {}

    def set_probe(self, probe):
        """Override to properly initialize streams_to_map"""
        super(PluginStreamMapper, self).set_probe(probe)

        # Initialize list of audio streams to process
        self.streams_to_map = []

        # Get the probe data and filter for audio streams
        probe_data = self.probe.get_probe()
        if not probe_data:
            logger.error("No probe data available")
            return

        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") == "audio":
                stream_id = stream.get("index")
                if stream_id is not None:
                    self.streams_to_map.append(stream_id)
                    logger.debug(f"Found audio stream at index {stream_id}")

        logger.debug(
            f"Initialized {len(self.streams_to_map)} audio streams for processing"
        )

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

        # Reset stream tracking
        self.streams_to_map = []
        self.streams_by_language.clear()
        self.stereo_streams_by_language.clear()

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
        """Modified to handle channel validation and constraints"""
        try:
            # Validate stream info
            analysis = AudioStreamInfo(stream_info)

            # Get appropriate encoder
            try:
                encoder = self.get_encoder_for_stream(analysis)
            except ValueError as e:
                logger.error(f"No suitable encoder found: {str(e)}")
                # Fallback to stream copy if we can't encode
                return {
                    "stream_mapping": ["-map", f"0:a:{stream_id}"],
                    "stream_encoding": [f"-c:a:{stream_id}", "copy"],
                }

            # Check channel constraints
            if encoder == "libfdk_aac" and analysis.channels > 6:
                logger.warning(
                    f"libfdk_aac limited to 6 channels, input has {analysis.channels}. "
                    "Audio will be downmixed."
                )
                # Force 5.1 for libfdk_aac
                target_channels = 6
            else:
                target_channels = analysis.channels

            mapping = self.build_stream_mapping(stream_info, stream_id)
            if not mapping or not mapping.get("stream_mapping"):
                logger.warning(f"Failed to build mapping for stream {stream_id}")
                # Fallback to copy
                return {
                    "stream_mapping": ["-map", f"0:a:{stream_id}"],
                    "stream_encoding": [f"-c:a:{stream_id}", "copy"],
                }

            # Add channel constraint if needed
            if target_channels != analysis.channels:
                mapping["stream_encoding"].extend(
                    [f"-ac:a:{stream_id}", str(target_channels)]
                )

            return mapping

        except Exception as e:
            logger.error(f"Error mapping stream {stream_id}: {str(e)}")
            # Safe fallback
            return {
                "stream_mapping": ["-map", f"0:a:{stream_id}"],
                "stream_encoding": [f"-c:a:{stream_id}", "copy"],
            }

    def get_ffmpeg_args(self) -> List[str]:
        """Modified to handle all streams correctly"""
        # Start with base command args
        args = [
            "-hide_banner",  # Cleaner output
            "-loglevel",
            "info",  # Informative logging
            "-i",
            self.input_file[0],  # Input file
            "-map",
            "0",  # Map all streams by default
            "-c",
            "copy",  # Copy all streams by default
            "-max_muxing_queue_size",
            "4096",  # Prevent queue overflow
        ]

        # Process audio streams in correct order
        output_args = []
        processed_languages = set()
        stereo_languages = set()

        # First pass: Process high-quality main streams
        logger.info("Processing main high-quality streams...")
        for stream_id in self.streams_to_map:
            stream_info = self.probe.get_stream_info(stream_id)
            analysis = AudioStreamInfo(stream_info)

            # Skip if we already processed a main stream for this language
            if analysis.language in processed_languages:
                logger.debug(
                    f"Skipping duplicate main stream for language: {analysis.language}"
                )
                continue

            # Build audio mapping for this stream
            mapping = self._build_audio_mapping(stream_info, stream_id)
            if mapping:
                # Remove default copy mapping for this stream
                output_args.extend(["-map", "-0:a:" + str(stream_id)])
                # Add our custom mapping
                output_args.extend(mapping["stream_mapping"])
                output_args.extend(mapping["stream_encoding"])
                processed_languages.add(analysis.language)

                logger.debug(
                    f"Added main stream mapping for language: {analysis.language}"
                )

        # Second pass: Add stereo/compatibility streams if needed
        logger.info("Processing stereo compatibility streams...")
        for stream_id in self.streams_to_map:
            stream_info = self.probe.get_stream_info(stream_id)
            analysis = AudioStreamInfo(stream_info)

            if (
                analysis.language not in stereo_languages
                and analysis.channels > 2
                and self._needs_stereo_version(analysis)
            ):
                stereo_mapping = self._create_stereo_downmix(analysis, stream_id)
                output_args.extend(stereo_mapping["stream_mapping"])
                output_args.extend(stereo_mapping["stream_encoding"])
                stereo_languages.add(analysis.language)

                logger.debug(
                    f"Added stereo stream mapping for language: {analysis.language}"
                )

        # Add all mappings and encoding options
        args.extend(output_args)

        # Add output file
        args.extend(["-y", self.output_file[0]])

        # Log the complete command for debugging
        logger.debug("FFmpeg command: {}".format(" ".join(args)))

        return args


def _build_audio_mapping(self, stream_info: dict, stream_id: int):
    """Helper to build audio stream mapping"""
    analysis = AudioStreamInfo(stream_info)
    needs_sample_rate_conversion = analysis.sample_rate > self.settings.get_setting(
        "target_sample_rate"
    )

    if not needs_sample_rate_conversion and not self.settings.get_setting(
        "force_processing"
    ):
        logger.debug(
            f"Stream {stream_id} doesn't need processing, will use default copy"
        )
        return None

    return self.custom_stream_mapping(stream_info, stream_id)

    def set_input_file(self, path):
        """Override to ensure input file is stored correctly"""
        if isinstance(path, str):
            self.input_file = [path]
        else:
            self.input_file = path

    def set_output_file(self, path):
        """Override to ensure output file is stored correctly"""
        if isinstance(path, str):
            self.output_file = [path]
        else:
            self.output_file = path

    def validate_encoder(self, encoder: str) -> bool:
        """New function to validate encoder availability"""
        try:
            # Run ffmpeg -encoders and check output
            import subprocess

            result = subprocess.run(
                ["ffmpeg", "-encoders"], capture_output=True, text=True
            )
            return encoder in result.stdout
        except Exception as e:
            logger.error(f"Failed to validate encoder {encoder}: {str(e)}")
            return False

    def get_encoder_for_stream(self, stream_info: AudioStreamInfo) -> str:
        """Modified to handle encoder validation and fallbacks"""
        # First choice encoders
        if stream_info.channels > 6:
            encoder = self.settings.get_setting("multichannel_71_encoder")  # opus
        elif stream_info.channels > 2:
            encoder = self.settings.get_setting("multichannel_51_encoder")  # libfdk_aac
        else:
            encoder = self.settings.get_setting("stereo_encoder")  # libfdk_aac

        # Validate encoder is available
        if not self.validate_encoder(encoder):
            logger.warning(
                f"Preferred encoder {encoder} not available, using fallbacks"
            )

            # Fallback chain
            if encoder == "libfdk_aac":
                fallbacks = ["aac", "libopus", "ac3"]
            elif encoder == "opus":
                # For >6 channels, we need opus
                logger.error(
                    f"Cannot process {stream_info.channels} channels without opus encoder"
                )
                raise ValueError(
                    f"No suitable encoder available for {stream_info.channels} channels"
                )

            # Try fallbacks
            for fallback in fallbacks:
                if self.validate_encoder(fallback):
                    logger.info(f"Using fallback encoder: {fallback}")
                    return fallback

            raise ValueError("No suitable encoder available")

        return encoder

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

    def streams_need_processing(self):
        """Determine if any streams need processing"""
        for stream in self.probe.get_probe()["streams"]:
            if stream["codec_type"] == "audio":
                logger.debug(f"""
                Audio stream found:
                    Codec: {stream.get('codec_name')}
                    Sample Rate: {stream.get('sample_rate')}
                    Channels: {stream.get('channels')}
                    Bit Rate: {stream.get('bit_rate')}
                """)

        return super().streams_need_processing()


def on_library_management_file_test(data):
    """
    Runner function - enables additional actions during the library management file tests.
    """
    # Get the path to the file
    abspath = data.get("path")

    # Configure settings object
    settings = Settings(library_id=data.get("library_id"))

    # Get file probe
    probe = Probe(logger, allowed_mimetypes=["video"])
    if not probe.file(abspath):
        # File probe failed, skip the rest of this test
        logger.debug(f"Failed to probe file '{abspath}'")
        return data

    # Get stream mapper
    mapper = PluginStreamMapper()
    mapper.set_settings(settings)
    mapper.set_probe(probe)

    try:
        # Analyze all streams
        probe_streams = probe.get_probe()["streams"]
        mapper.analyze_streams(probe_streams)

        # Check if any streams need processing
        needs_processing = mapper.streams_need_processing()
        logger.debug(f"File '{abspath}' needs processing: {needs_processing}")

        if needs_processing:
            data["add_file_to_pending_tasks"] = True
            logger.info(
                f"File '{abspath}' added to task list - streams require processing"
            )
        else:
            logger.debug(f"File '{abspath}' does not require processing")

    except Exception as e:
        logger.error(f"Error analyzing file '{abspath}': {str(e)}")
        # Don't add file if we encounter an error
        return data

    return data


def on_worker_process(data):
    """
    Runner function - enables additional configured processing jobs during the worker stages of a task.

    The 'data' object argument includes:
        worker_log              - Array, the log lines that are being tailed by the frontend. Can be left empty.
        library_id             - Number, the library that the current task is associated with.
        exec_command           - Array, a subprocess command that Unmanic should execute. Can be empty.
        command_progress_parser - Function, a function that Unmanic can use to parse the STDOUT of the command to collect progress stats. Can be empty.
        file_in               - String, the source file to be processed by the command.
        file_out              - String, the destination that the command should output (may be the same as the file_in if necessary).
        original_file_path    - String, the absolute path to the original file.
        repeat               - Boolean, should this runner be executed again once completed with the same variables.
    """
    # Get the path to the file
    abspath = data.get("file_in")

    # Configure settings object
    settings = Settings(library_id=data.get("library_id"))

    # Get file probe
    probe = Probe(logger, allowed_mimetypes=["video"])
    if not probe.file(abspath):
        # File probe failed, skip the rest of this test
        logger.debug(f"Failed to probe file '{abspath}'")
        return data

    # Get stream mapper
    mapper = PluginStreamMapper()
    mapper.set_settings(settings)
    mapper.set_probe(probe)

    try:
        # Analyze all streams
        probe_streams = probe.get_probe()["streams"]
        mapper.analyze_streams(probe_streams)

        # Check if any streams need processing
        if mapper.streams_need_processing():
            # Set input/output files
            mapper.set_input_file(data.get("file_in"))
            mapper.set_output_file(data.get("file_out"))

            # Get the FFmpeg args
            ffmpeg_args = mapper.get_ffmpeg_args()

            if ffmpeg_args:
                # Set the execution command
                data["exec_command"] = ["ffmpeg"]
                data["exec_command"].extend(ffmpeg_args)

                # Set up the parser
                data["command_progress_parser"] = Parser(logger)

                logger.debug(
                    "FFmpeg command generated: {}".format(
                        " ".join(data["exec_command"])
                    )
                )
            else:
                logger.debug("No FFmpeg command required")
        else:
            logger.debug("No streams need processing")

    except Exception as e:
        logger.error(f"Error during worker processing: {str(e)}")
        # Don't process if we encounter an error
        return data

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
