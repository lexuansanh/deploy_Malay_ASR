from webrtc import WebRTC
from herpetologist import check_type


@check_type
def webrtc(
    aggressiveness: int = 3,
    sample_rate: int = 16000,
    minimum_amplitude: int = 100,
):
    """
    Load WebRTC VAD model.
    Parameters
    ----------
    aggressiveness: int, optional (default=3)
        an integer between 0 and 3.
        0 is the least aggressive about filtering out non-speech, 3 is the most aggressive.
    sample_rate: int, optional (default=16000)
        sample rate for samples.
    minimum_amplitude: int, optional (default=100)
        abs(minimum_amplitude) to assume a sample is a voice activity. Else, automatically False.
    Returns
    -------
    result : malaya_speech.model.webrtc.WebRTC class
    """

    try:
        import webrtcvad
    except BaseException:
        raise ModuleNotFoundError(
            'webrtcvad not installed. Please install it by `pip install webrtcvad` and try again.'
        )

    vad = webrtcvad.Vad(aggressiveness)
    return WebRTC(vad, sample_rate, minimum_amplitude)