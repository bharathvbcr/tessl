//! The figures `plot_results` knows: a name, the result file each is drawn
//! from, and the function that draws it.

use crate::data::R;
use std::path::Path;

/// What a figure produces: the SVG document and any table its script also
/// printed (the audio profile's grouped table, which its results `.txt` quotes).
pub struct Output {
    pub svg: String,
    pub table: String,
}

impl Output {
    pub fn svg(svg: String) -> Output {
        Output {
            svg,
            table: String::new(),
        }
    }
}

pub struct Plot {
    pub name: &'static str,
    /// The result file the figure is drawn from, in the results directory.
    pub file: &'static str,
    pub render: fn(&Path) -> R<Output>,
}

pub const PLOTS: &[Plot] = &[
    Plot {
        name: "final",
        file: "embedgemma2_final_m5pro.json",
        render: crate::paired::final_,
    },
    Plot {
        name: "audio-speed",
        file: "embedgemma2_audio_speed_m5pro.json",
        render: crate::paired::audio_speed,
    },
    Plot {
        name: "mm-onewait",
        file: "embedgemma2_mm_onewait_m5pro.json",
        render: crate::paired::mm_onewait,
    },
    Plot {
        name: "pre",
        file: "embedgemma2_pre_m5pro.json",
        render: crate::paired::pre,
    },
    Plot {
        name: "vision-speed",
        file: "embedgemma2_vision_speed_m5pro.json",
        render: crate::vision::vision_speed,
    },
    Plot {
        name: "audio-profile",
        file: "embedgemma2_audio_profile_m5pro.json",
        render: crate::stage::audio_profile,
    },
    Plot {
        name: "stage-profile",
        file: "embedgemma2_stage_profile_m5pro.json",
        render: crate::stage::stage_profile,
    },
    Plot {
        name: "text-fusion",
        file: "embedgemma2_text_fusion_profile_m5pro.json",
        render: crate::stage::text_fusion,
    },
    Plot {
        name: "memory",
        file: "embedgemma2_memory_m5pro.json",
        render: crate::memory::memory,
    },
    Plot {
        name: "small-m",
        file: "embedgemma2_small_m_m5pro.json",
        render: crate::small_m::small_m,
    },
    Plot {
        name: "tok",
        file: "embedgemma2_tok_m5pro.json",
        render: crate::tok::tok,
    },
    Plot {
        name: "video",
        file: "embedgemma2_video_m5pro.json",
        render: crate::tok::video,
    },
    Plot {
        name: "media-decode",
        file: "media_decode_m5pro.json",
        render: crate::media::media_decode,
    },
    Plot {
        name: "media-resample",
        file: "media_resample_m5pro.json",
        render: crate::resample::resample,
    },
    Plot {
        name: "video-color",
        file: "media_video_color_m5pro.json",
        render: crate::video_color::video_color,
    },
    Plot {
        name: "jpeg-decode",
        file: "jpeg_decode_m5pro.json",
        render: crate::jpeg::jpeg_decode,
    },
];

pub fn find(name: &str) -> Option<&'static Plot> {
    PLOTS.iter().find(|p| p.name == name)
}

pub fn names() -> String {
    PLOTS.iter().map(|p| p.name).collect::<Vec<_>>().join(", ")
}
