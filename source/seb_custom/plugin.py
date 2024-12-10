#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from seb_custom.lib.ffmpeg import Parser, Probe, StreamMapper
from unmanic.libs.unplugins.settings import PluginSettings

# Configure plugin logger
logger = logging.getLogger("Unmanic.Plugin.seb_custom")
logger.setLevel(logging.DEBUG)


class Settings(PluginSettings):
    settings = {
        "max_sample_rate": 48000,  # Default max supported sample rate
    }

    def __init__(self, *args, **kwargs):
        super(Settings, self).__init__(*args, **kwargs)
        self.form_settings = {
            "max_sample_rate": {
                "label": "Maximum supported audio sample rate (Hz)",
                "input_type": "slider",
                "slider_options": {"min": 22050, "max": 192000, "step": 50},
                "help_text": "Any audio stream above this sample rate will be re-encoded down to this rate.",
            }
        }


class PluginStreamMapper(StreamMapper):
    def __init__(self):
        super(PluginStreamMapper, self).__init__(logger, ["audio"])
        self.codec = "aac"
        self.encoder = "libfdk_aac"
        self.settings = None

    def set_default_values(self, settings, abspath, probe):
        logger.debug(f"Setting default values for file '{abspath}'")
        self.abspath = abspath
        self.set_probe(probe)
        self.set_input_file(abspath)
        self.settings = settings
        logger.debug("Default values set.")

    @staticmethod
    def calculate_bitrate(stream_info: dict):
        # Default logic: 64 kb/s per channel
        channels = stream_info.get("channels", 2)
        return int(channels) * 64

    def test_stream_needs_processing(self, stream_info: dict):
        # Retrieve max_sample_rate from settings
        value = self.settings.get_setting("max_sample_rate")
        if value is None:
            value = 48000
        max_rate = int(value)

        sample_rate = int(stream_info.get("sample_rate", 0))
        codec_name = stream_info.get("codec_name", "unknown")
        channels = stream_info.get("channels", "unknown")

        logger.debug(
            f"Testing stream (codec={codec_name}, sample_rate={sample_rate}, "
            f"channels={channels}, max_rate={max_rate}) for processing."
        )

        # Re-encode if sample rate is above max_rate
        if sample_rate > max_rate:
            logger.debug(
                f"Stream sample rate {sample_rate} > {max_rate} Hz. Will re-encode."
            )
            return True

        logger.debug(
            f"Stream sample rate {sample_rate} <= {max_rate} Hz. No processing needed."
        )
        return False

    def custom_stream_mapping(self, stream_info: dict, stream_id: int):
        # This is only called if test_stream_needs_processing() returned True
        value = self.settings.get_setting("max_sample_rate")
        if value is None:
            value = 48000
        max_rate = int(value)

        calculated_bitrate = self.calculate_bitrate(stream_info)
        channels = int(stream_info.get("channels", 2))
        if channels > 6:
            channels = 6

        logger.debug(
            f"Custom mapping for stream {stream_id}: "
            f"re-encode to {self.encoder}, {channels} channels, {calculated_bitrate}k, {max_rate} Hz."
        )

        # Build the encoding args for this stream
        stream_encoding = [
            "-c:a:{}".format(stream_id),
            self.encoder,
            "-ac:a:{}".format(stream_id),
            "{}".format(channels),
            "-b:a:{}".format(stream_id),
            "{}k".format(calculated_bitrate),
            "-ar:a:{}".format(stream_id),
            str(max_rate),
        ]

        return {
            "stream_mapping": ["-map", f"0:a:{stream_id}"],
            "stream_encoding": stream_encoding,
        }


def on_library_management_file_test(data):
    abspath = data.get("path")
    logger.debug(f"on_library_management_file_test called for '{abspath}'")

    probe = Probe(logger, allowed_mimetypes=["audio", "video"])
    if not probe.file(abspath):
        logger.warning(f"File probe failed for '{abspath}'. Skipping test.")
        return data

    # Load settings for this library (or global if no library_id)
    settings = (
        Settings(library_id=data.get("library_id"))
        if data.get("library_id")
        else Settings()
    )

    mapper = PluginStreamMapper()
    mapper.set_default_values(settings, abspath, probe)

    if mapper.streams_need_processing():
        data["add_file_to_pending_tasks"] = True
        logger.debug(
            f"File '{abspath}' requires processing and has been added to the pending tasks."
        )
    else:
        logger.debug(f"File '{abspath}' does not require processing.")

    return data


def on_worker_process(data):
    abspath = data.get("file_in")
    logger.debug(f"on_worker_process called for '{abspath}'")

    data["exec_command"] = []
    data["repeat"] = False

    probe = Probe(logger, allowed_mimetypes=["audio", "video"])
    if not probe.file(abspath):
        logger.warning(f"File probe failed for '{abspath}'. Cannot process.")
        return data

    settings = Settings(library_id=data.get("library_id"))

    mapper = PluginStreamMapper()
    mapper.set_default_values(settings, abspath, probe)

    if mapper.streams_need_processing():
        logger.debug(f"Preparing ffmpeg command for '{abspath}'.")

        mapper.set_input_file(abspath)
        mapper.set_output_file(data.get("file_out"))

        # Get the ffmpeg args from the mapper
        ffmpeg_args = mapper.get_ffmpeg_args()

        # Insert `-strict -2` if needed to enable libfdk_aac
        # and ensure a large muxing queue size like in the original code.
        # The original code placed `-strict -2` and `-max_muxing_queue_size` before mapping.
        # We'll do the same for consistency.

        # Typically, StreamMapper already sets -max_muxing_queue_size (if it was included),
        # but let's ensure it here if needed. The original code included it, so let's add it back.
        # You can adjust this or remove if unnecessary.
        if "-max_muxing_queue_size" not in ffmpeg_args:
            ffmpeg_args.insert(0, "-max_muxing_queue_size")
            ffmpeg_args.insert(1, "4096")

        if "-strict" not in ffmpeg_args:
            ffmpeg_args.insert(0, "-strict")
            ffmpeg_args.insert(1, "-2")

        # Also include `-hide_banner -loglevel info` at the start if desired.
        if "-hide_banner" not in ffmpeg_args:
            ffmpeg_args.insert(0, "-loglevel")
            ffmpeg_args.insert(1, "info")
            ffmpeg_args.insert(0, "-hide_banner")

        # Construct the final ffmpeg command
        data["exec_command"] = ["ffmpeg"] + ffmpeg_args

        logger.debug(f"FFmpeg command: {' '.join(data['exec_command'])}")

        parser = Parser(logger)
        parser.set_probe(probe)
        data["command_progress_parser"] = parser.parse_progress
    else:
        logger.debug(
            f"No processing needed for '{abspath}'. No ffmpeg command will be run."
        )

    return data
