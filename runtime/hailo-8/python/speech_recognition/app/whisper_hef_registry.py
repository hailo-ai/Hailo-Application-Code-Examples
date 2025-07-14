import os

HEF_REGISTRY = {
    "base": {
        "hailo8": {
            "encoder": "app/hefs/h8/base/base-whisper-encoder-5s.hef",
            "decoder": "app/hefs/h8/base/base-whisper-decoder-fixed-sequence-matmul-split.hef",
        },
        "hailo8l": {
            "encoder": "app/hefs/h8l/base/base-whisper-encoder-5s_h8l.hef",
            "decoder": "app/hefs/h8l/base/base-whisper-decoder-fixed-sequence-matmul-split_h8l.hef",
        }
    },
    "tiny": {
        "hailo8": {
            "encoder": "app/hefs/h8/tiny/tiny-whisper-encoder-10s_15dB.hef",
            "decoder": "app/hefs/h8/tiny/tiny-whisper-decoder-fixed-sequence-matmul-split.hef",
        },
        "hailo8l": {
            "encoder": "app/hefs/h8l/tiny/tiny-whisper-encoder-10s_15dB_h8l.hef",
            "decoder": "app/hefs/h8l/tiny/tiny-whisper-decoder-fixed-sequence-matmul-split_h8l.hef",
        }
    }
}