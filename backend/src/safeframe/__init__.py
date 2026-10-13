"""SafeFrame: blur people estimated to be under 18 in video.

Age estimation is probabilistic. SafeFrame is designed to fail closed: anyone
it cannot confidently classify as an adult is blurred, and every result goes
through a human review step before export.
"""

__version__ = "0.1.0"
