// Every module of `plot_results`, listed once. `main.rs` and each test that
// drives the figures (`tests/plot_results*.rs`) `include!` this file, so a new
// figure's module is declared in one place and no test can miss it (the
// registry names every figure's module, so a list that missed one would not
// compile). Paths are relative to this file. The tests do not reach every
// item, hence the `dead_code` allowances.

// The crate's strict parser, shared the way bench_paired shares it; this tool
// does not use its object lookup.
#[allow(dead_code)]
#[path = "../../json.rs"]
mod json;

#[allow(dead_code)]
#[path = "chart.rs"]
mod chart;
#[allow(dead_code)]
#[path = "data.rs"]
mod data;
#[allow(dead_code)]
#[path = "go_overhead.rs"]
mod go_overhead;
#[allow(dead_code)]
#[path = "jpeg.rs"]
mod jpeg;
#[allow(dead_code)]
#[path = "media.rs"]
mod media;
#[allow(dead_code)]
#[path = "memory.rs"]
mod memory;
#[allow(dead_code)]
#[path = "paired.rs"]
mod paired;
#[allow(dead_code)]
#[path = "registry.rs"]
mod registry;
#[allow(dead_code)]
#[path = "resample.rs"]
mod resample;
#[allow(dead_code)]
#[path = "small_m.rs"]
mod small_m;
#[allow(dead_code)]
#[path = "stage.rs"]
mod stage;
#[allow(dead_code)]
#[path = "svg.rs"]
mod svg;
#[allow(dead_code)]
#[path = "tok.rs"]
mod tok;
#[allow(dead_code)]
#[path = "video_color.rs"]
mod video_color;
#[allow(dead_code)]
#[path = "vision.rs"]
mod vision;
