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
        },
        "hailo10h": {
            "encoder": "app/hefs/h10h/tiny/tiny-whisper-encoder-10s.hef",
            "decoder": "app/hefs/h10h/tiny/tiny-whisper-decoder-fixed-sequence.hef",
        }
    },
    "tiny.en": {
        "hailo10h": {
                "encoder": "app/hefs/h10h/tiny.en/tiny_en-whisper-encoder-10s.hef",
                "decoder": "app/hefs/h10h/tiny.en/tiny_en-whisper-decoder-fixed-sequence.hef",
        }
    }
}