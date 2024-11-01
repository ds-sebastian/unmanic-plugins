
---

##### Links:

- [Support](https://unmanic.app/discord)
- [Issues/Feature Requests](https://github.com/Unmanic/plugin.encoder_audio_aac/issues)
- [Pull Requests](https://github.com/Unmanic/plugin.encoder_audio_aac/pulls)

---

Here's a comprehensive description.md for our audio processing plugin:

##### Description:

This plugin provides comprehensive audio stream management for your media files, ensuring optimal compatibility with modern devices and streaming services while maintaining quality.

Key features:
- Automatically manages sample rates (converts to 48kHz when needed)
- Intelligently handles multichannel audio (5.1, 7.1)
- Creates stereo downmixes of multichannel content when appropriate
- Applies volume normalization separately for stereo and surround content
- Preserves all language tracks and metadata
- Optimizes bitrates based on channel count

The plugin follows best practices for audio processing:
- Uses libfdk_aac for stereo/5.1 content (64 kbit/s per channel)
- Automatically switches to OPUS for 7.1 content to preserve all channels
- Applies professional loudness standards (-24 LUFS for surround, -16 LUFS for stereo)
- Preserves original streams when no processing is needed

---

##### Documentation:

For information on the encoders and standards used:
- [FFmpeg - AAC Encoding](https://trac.ffmpeg.org/wiki/Encode/AAC)
- [FFmpeg - OPUS Encoding](https://trac.ffmpeg.org/wiki/Encode/Opus)
- [ITU-R BS.1770-4 Loudness Normalization](https://www.itu.int/rec/R-REC-BS.1770)

---

### Config description:

#### <span style="color:blue">Keep Original Streams</span>
When enabled, preserves the original audio streams alongside any processed versions. Useful for maintaining maximum compatibility or quality where storage space isn't a concern.

#### <span style="color:blue">Always Create Stereo</span>
Creates stereo versions of multichannel audio streams. Particularly useful for ensuring compatibility with devices that don't support surround sound.

#### <span style="color:blue">Stereo Encoder</span>
Choose between libfdk_aac (better quality) or native AAC (wider compatibility) for stereo streams.

#### <span style="color:blue">Loudness Normalization Settings</span>
Separate controls for stereo and surround content:
- **Integrated Loudness (I)**: Target loudness level (LUFS)
- **Loudness Range (LRA)**: Acceptable variation in loudness
- **True Peak (TP)**: Maximum allowed peak level

Recommended values:
- Surround: I=-24.0, LRA=7.0, TP=-2.0
- Stereo: I=-16.0, LRA=11.0, TP=-1.0

#### <span style="color:blue">Max Muxing Queue Size</span>
Sets the FFmpeg packet buffer size. Increase if you encounter muxing errors with complex audio streams.

:::note
Bitrates are automatically calculated using best practices:
- Stereo: 128 kbit/s (64 kbit/s per channel)
- 5.1: 384 kbit/s (64 kbit/s per channel)
- 7.1: Handled by OPUS with appropriate bitrate
:::