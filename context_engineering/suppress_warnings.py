import logging
import os
import sys
import warnings


class StderrFilter:
    """Intercepts and filters out raw stderr messages from third-party SDKs."""

    def __init__(self, original_stderr):
        self.original_stderr = original_stderr

    def write(self, s):
        if "automatic function calling" in s or "AFC" in s:
            return
        self.original_stderr.write(s)

    def flush(self):
        self.original_stderr.flush()


def silence_warnings():
    """Completely silences Google SDK AFC warnings across all output channels."""
    # Disable Python warnings globally
    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.simplefilter("ignore")

    # 2. Quiet down SDK loggers
    for logger_name in [
        "google",
        "google.genai",
        "google.genai.models",
        "langchain_google_genai",
        "absl",
    ]:
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

    try:
        import absl.logging                                                                                            # type: ignore  # noqa: I001

        absl.logging.set_verbosity(absl.logging.CRITICAL)
    except Exception:                                                                                                                        # noqa 
        pass

    # intercept raw sys.stderr writes
    if not isinstance(sys.stderr, StderrFilter):
        sys.stderr = StderrFilter(sys.stderr)
