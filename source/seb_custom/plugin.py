#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    unmanic-plugins.plugin.py

    Written by: Sebastian 
    Date: October 31, 2024

    A comprehensive audio processing plugin that ensures optimal compatibility 
    with modern devices by managing sample rates, channel counts, and volume 
    levels while preserving quality and metadata.

    Features:
    - Converts high sample rates (>48kHz) to 48kHz
    - Creates stereo versions of multichannel content
    - Uses OPUS for >6 channels, libfdk_aac otherwise
    - Preserves original streams when possible
    - Maintains all language tracks
"""
import logging
import os

from unmanic.libs.unplugins.settings import PluginSettings
from unmanic.libs.directoryinfo import UnmanicDirectoryInfo

from seb_custom.lib.ffmpeg import StreamMapper, Probe, Parser

# Configure plugin logger
logger = logging.getLogger("Unmanic.Plugin.seb_custom")

class Settings(PluginSettings):
    settings = {}  # No configurable settings - everything is hardcoded

class PluginStreamMapper(StreamMapper):
    def __init__(self):
        super(PluginStreamMapper, self).__init__(logger, ['audio'])
        self.target_sample_rate = 48000
        self.stereo_streams_by_language = {}  # Track existing stereo streams
        self.stream_count = 0  # Track output stream count for proper mapping

    def analyze_stream(self, stream_info: dict) -> dict:
        """Analyze single stream and return relevant info"""
        try:
            return {
                'index': stream_info.get('index', 0),
                'codec': stream_info.get('codec_name', '').lower(),
                'channels': int(stream_info.get('channels', 2)),
                'sample_rate': int(stream_info.get('sample_rate', 48000)),
                'language': stream_info.get('tags', {}).get('language', 'und'),
                'title': stream_info.get('tags', {}).get('title', ''),
                'bitrate': int(stream_info.get('bit_rate', 128000)),
                'is_atmos': 'atmos' in str(stream_info).lower()
            }
        except Exception as e:
            logger.error(f"Stream analysis error: {str(e)}")
            return None
        
    def test_stream_needs_processing(self, stream_info: dict) -> bool:
            """Determine if stream needs processing"""
            analysis = self.analyze_stream(stream_info)
            if not analysis:
                return False

            logger.info(f"""
    Stream Analysis:
        Index: {analysis['index']}
        Codec: {analysis['codec']}
        Channels: {analysis['channels']}
        Sample Rate: {analysis['sample_rate']} Hz
        Language: {analysis['language']}
        Title: {analysis['title']}
        Is Atmos: {analysis['is_atmos']}
    """)

            # Must convert if sample rate too high
            if analysis['sample_rate'] > self.target_sample_rate:
                logger.info(f"Stream requires processing: {analysis['sample_rate']}Hz > {self.target_sample_rate}Hz target")
                return True

            # Create stereo version if multichannel and no stereo exists for this language
            if analysis['channels'] > 2 and analysis['language'] not in self.stereo_streams_by_language:
                logger.info(f"Stream requires processing: Need stereo version for {analysis['language']}")
                return True

            logger.info(f"Stream does not require processing: Compatible sample rate and stereo version exists")
            return False

        def calculate_bitrate(self, channels: int) -> str:
            """Calculate bitrate based on channels (64k per channel)"""
            return f"{64 * channels}k"

        def custom_stream_mapping(self, stream_info: dict, stream_id: int) -> dict:
            """Build stream mapping command"""
            analysis = self.analyze_stream(stream_info)
            if not analysis:
                # If analysis fails, just copy the stream
                return {
                    'stream_mapping': ['-map', f'0:a:{stream_id}'],
                    'stream_encoding': [f'-c:a:{self.stream_count}', 'copy']
                }

            stream_mapping = []
            stream_encoding = []

            # First handle the main stream
            needs_sample_rate_conversion = analysis['sample_rate'] > self.target_sample_rate

            if needs_sample_rate_conversion:
                # Need to re-encode - choose encoder based on channels
                if analysis['channels'] > 6:
                    encoder = 'libopus'
                    logger.info(f"Using OPUS for {analysis['channels']} channels")
                else:
                    encoder = 'libfdk_aac'
                    logger.info(f"Using libfdk_aac for {analysis['channels']} channels")

                stream_mapping.extend(['-map', f'0:a:{stream_id}'])
                stream_encoding.extend([
                    f'-c:a:{self.stream_count}', encoder,
                    f'-ar:a:{self.stream_count}', str(self.target_sample_rate),
                    f'-b:a:{self.stream_count}', self.calculate_bitrate(analysis['channels'])
                ])

                # Preserve metadata
                if analysis['language'] != 'und':
                    stream_encoding.extend([
                        f'-metadata:s:a:{self.stream_count}', f'language={analysis["language"]}'
                    ])
                if analysis['title']:
                    stream_encoding.extend([
                        f'-metadata:s:a:{self.stream_count}', f'title={analysis["title"]}'
                    ])
            else:
                # Just copy the stream if no conversion needed
                stream_mapping.extend(['-map', f'0:a:{stream_id}'])
                stream_encoding.extend([f'-c:a:{self.stream_count}', 'copy'])

            self.stream_count += 1

            # Create stereo version if needed
            if (analysis['channels'] > 2 and 
                analysis['language'] not in self.stereo_streams_by_language):
                
                # Add stereo stream mapping
                stream_mapping.extend(['-map', f'0:a:{stream_id}'])
                stream_encoding.extend([
                    f'-c:a:{self.stream_count}', 'libfdk_aac',
                    f'-ac:a:{self.stream_count}', '2',
                    f'-ar:a:{self.stream_count}', str(self.target_sample_rate),
                    f'-b:a:{self.stream_count}', '128k',
                    f'-metadata:s:a:{self.stream_count}', f'title=Stereo',
                ])
                
                # Add language metadata to stereo stream
                if analysis['language'] != 'und':
                    stream_encoding.extend([
                        f'-metadata:s:a:{self.stream_count}', f'language={analysis["language"]}'
                    ])

                self.stereo_streams_by_language[analysis['language']] = True
                self.stream_count += 1
                logger.info(f"Created stereo version for language: {analysis['language']}")

            return {
                'stream_mapping': stream_mapping,
                'stream_encoding': stream_encoding
            }
    def on_library_management_file_test(data):
        """
        Runner function - enables additional actions during the library management file tests.

        The 'data' object argument includes:
            path                            - String containing the full path to the file being tested.
            issues                          - List of currently found issues for not processing the file.
            add_file_to_pending_tasks       - Boolean, is the file currently marked to be added to the queue for processing.
        """
        # Get the path to the file
        abspath = data.get('path')

        # Get file probe
        probe = Probe(logger, allowed_mimetypes=['video'])
        if not probe.file(abspath):
            # File probe failed, skip the rest of this test
            logger.debug(f"Failed to probe file: {abspath}")
            return data

        try:
            # Get stream mapper
            mapper = PluginStreamMapper()
            mapper.set_probe(probe)

            # Check audio streams
            audio_streams = [s for s in probe.get_probe()['streams'] 
                            if s.get('codec_type') == 'audio']
            
            if not audio_streams:
                logger.debug(f"No audio streams found in {abspath}")
                return data

            logger.debug(f"Found {len(audio_streams)} audio streams in {abspath}")

            # Check if any stream needs processing
            needs_processing = False
            for stream in audio_streams:
                if mapper.test_stream_needs_processing(stream):
                    needs_processing = True
                    break

            if needs_processing:
                data['add_file_to_pending_tasks'] = True
                logger.info(f"File requires audio processing: {abspath}")
            else:
                logger.debug(f"No audio processing needed: {abspath}")

        except Exception as e:
            logger.error(f"Error testing file: {abspath} - {str(e)}")

        return data


    def on_worker_process(data):
        """
        Runner function - enables additional configured processing jobs during the worker stages of a task.

        The 'data' object argument includes:
            exec_command            - A command that Unmanic should execute. Can be empty.
            command_progress_parser - A function that Unmanic can use to parse the STDOUT of the command to collect progress stats. Can be empty.
            file_in                - The source file to be processed by the command.
            file_out               - The destination that the command should output (may be the same as the file_in if necessary).
            original_file_path     - The absolute path to the original file.
            repeat                - Boolean, should this runner be executed again once completed with the same variables.
        """
        # Default to no FFMPEG command required
        data['exec_command'] = []
        data['repeat'] = False

        # Get the path to the file
        abspath = data.get('file_in')

        try:
            # Get file probe
            probe = Probe(logger, allowed_mimetypes=['video'])
            if not probe.file(abspath):
                logger.error(f"Failed to probe file: {abspath}")
                return data

            # Get stream mapper
            mapper = PluginStreamMapper()
            mapper.set_probe(probe)

            # Check if any streams need processing
            if not any(mapper.test_stream_needs_processing(s) for s in probe.get_probe()['streams'] 
                    if s.get('codec_type') == 'audio'):
                logger.debug("No streams need processing")
                return data

            # Set the input/output files
            mapper.set_input_file(abspath)
            mapper.set_output_file(data.get('file_out'))

            # Build ffmpeg args
            ffmpeg_args = [
                '-hide_banner',
                '-loglevel', 'info',
                '-i', abspath,
                # Map all streams
                '-map', '0:v',   # Video
                '-c:v', 'copy',  # Copy video
                '-map', '0:s?',  # Subtitles (if present)
                '-c:s', 'copy',  # Copy subtitles
                '-map_chapters', '0'  # Keep chapters
            ]

            # Process all audio streams
            audio_streams = [s for s in probe.get_probe()['streams'] 
                            if s.get('codec_type') == 'audio']
            
            for stream in audio_streams:
                stream_id = stream.get('index')
                mapping = mapper.custom_stream_mapping(stream, stream_id)
                if mapping:
                    ffmpeg_args.extend(mapping['stream_mapping'])
                    ffmpeg_args.extend(mapping['stream_encoding'])

            # Add output file
            ffmpeg_args.extend([
                '-max_muxing_queue_size', '4096',
                '-y',
                data.get('file_out')
            ])

            # Set the execution command
            data['exec_command'] = ['ffmpeg']
            data['exec_command'].extend(ffmpeg_args)

            # Set up the parser
            parser = Parser(logger)
            parser.set_probe(probe)
            data['command_progress_parser'] = parser.parse_progress

            logger.debug(f"FFmpeg command: {' '.join(data['exec_command'])}")

        except Exception as e:
            logger.error(f"Error processing file: {abspath} - {str(e)}")

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
        if not data.get('task_processing_success'):
            logger.debug("Task processing was not successful")
            return data

        try:
            # Record processing history
            for destination_file in data.get('destination_files', []):
                directory_info = UnmanicDirectoryInfo(os.path.dirname(destination_file))
                
                # Get file probe to verify results
                probe = Probe(logger, allowed_mimetypes=['video'])
                if probe.file(destination_file):
                    audio_streams = [s for s in probe.get_probe()['streams'] 
                                if s.get('codec_type') == 'audio']
                    
                    info = {
                        'version': '1.0',
                        'processed_date': str(datetime.datetime.now()),
                        'audio_streams': len(audio_streams),
                        'stream_details': [{
                            'codec': s.get('codec_name'),
                            'channels': s.get('channels'),
                            'sample_rate': s.get('sample_rate'),
                            'language': s.get('tags', {}).get('language'),
                            'title': s.get('tags', {}).get('title')
                        } for s in audio_streams]
                    }
                    
                    directory_info.set('seb_custom', os.path.basename(destination_file), info)
                    directory_info.save()
                    
                    logger.debug(f"Saved processing info for {destination_file}")
                    logger.debug(f"Final audio configuration: {info}")

        except Exception as e:
            logger.error(f"Error in post-processing: {str(e)}")

        return data