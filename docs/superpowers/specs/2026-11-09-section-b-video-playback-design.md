# Section B Video Playback Design

## Intent

Make processed dashcam videos play in Chrome. The current MP4 output uses the
`mp4v`/FMP4 codec, which OpenCV can decode but Chrome cannot play.

## Goals

- Write processed videos in browser-compatible WebM/VP8 format.
- Point Streamlit's video player and the displayed output path to the generated
  `.webm` file.
- Report a clear error if the VP8 writer cannot be initialized.
- Preserve frame sampling, segmentation, telemetry, and inference behavior.

## Non-goals and constraints

- Do not change model inference, accuracy, input video support, or frame skipping.
- Do not add FFmpeg or another runtime dependency.
- The current OpenCV installation supports writing VP8/WebM.

## Approaches

- **Selected: OpenCV WebM/VP8 output.** Uses the available encoder directly and
  avoids an extra dependency or a second transcode pass.
- **FFmpeg H.264 output.** Broadly compatible, but requires installing and
  maintaining an additional runtime dependency.

## Design

Update the video processor to create a `.webm` output with OpenCV's VP8 codec,
and fail explicitly if the writer cannot initialize. Update the Section B output
filename so its extension matches the encoded format. Pass the processor's
returned path as a `video/webm` data URI to `st.video`. Streamlit's default MP4
MIME type is incorrect for WebM, and its `/media` endpoint returns HTTP 500 for
the browser's byte-range request in the current environment. Add focused
regression coverage for WebM writer output, frame sampling, output naming, and
lossless data-URI encoding. On skipped inference frames, blend the current
source frame with the most recent prediction so the output preserves motion
without changing inference frequency or telemetry counts.
