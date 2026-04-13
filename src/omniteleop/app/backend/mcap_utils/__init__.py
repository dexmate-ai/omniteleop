"""
MCAP-based recording and reading for teleoperation data.

This module provides production-ready MCAP recording functionality:
- TeleopMcapWriter: Write teleop episodes to MCAP files with protobuf serialization
- TeleopMcapReader: Read teleop episodes from MCAP files
- Episode: Container for episode data read from MCAP
- EpisodeMetadata: Configuration for episode recording

Usage:
    from omniteleop.record.mcap_utils import TeleopMcapWriter, TeleopMcapReader, EpisodeMetadata, Episode

Before using this module, compile the protobuf schemas:
    cd omniteleop/src/omniteleop/record/mcap_utils && ./compile_proto.sh
"""

from .metadata import EpisodeMetadata, SCHEMA_VERSION

from .mcap_writer import TeleopMcapWriter

from .mcap_reader import TeleopMcapReader

from .episode import Episode, PolicyStepAnnotation

__all__ = [
    # Config
    "EpisodeMetadata",
    "SCHEMA_VERSION",
    # Writer
    "TeleopMcapWriter",
    # Reader
    "TeleopMcapReader",
    "Episode",
    "PolicyStepAnnotation",
]
