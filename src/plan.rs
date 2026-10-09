//! Pure planning: `VideoInfo` + options → target geometry, fps and filter chain.
use crate::probe::VideoInfo;

#[derive(Debug, Clone)]
pub struct PlanOpts {
    pub height: u32,
    pub max_fps: f64,
    pub sharpen: bool,
}

impl Default for PlanOpts {
    fn default() -> Self {
        Self {
            height: 720,
            max_fps: 24.0,
            sharpen: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Geometry {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
}

/// Size as displayed to the viewer (rotation and SAR applied).
pub fn display_size(i: &VideoInfo) -> (f64, f64) {
    let w = i.width as f64 * if i.sar > 0.0 { i.sar } else { 1.0 };
    let h = i.height as f64;
    if i.rotation % 180 == 90 {
        (h, w)
    } else {
        (w, h)
    }
}

fn even_floor(v: f64) -> u32 {
    ((v.floor() as u32) & !1).max(2)
}

/// Fit inside the box without ever upscaling; both sides even.
/// Landscape/square: `height * 16/9` x `height`. Portrait: width unbounded, `height` tall.
pub fn geometry(i: &VideoInfo, o: &PlanOpts) -> Geometry {
    let (dw, dh) = display_size(i);
    let box_h = o.height as f64;
    let box_w = if dw > dh {
        (box_h * 16.0 / 9.0).round()
    } else {
        f64::MAX
    };
    let scale = (box_w / dw).min(box_h / dh).min(1.0);
    Geometry {
        width: even_floor(dw * scale),
        height: even_floor(dh * scale),
        fps: if i.fps > o.max_fps { o.max_fps } else { i.fps },
    }
}

/// HDR (PQ/HLG) → SDR bt709 using zscale + tonemap.
pub const TONEMAP: &str = "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p";

/// CPU-side scale/sharpen part shared by the sw and hw paths.
/// ffmpeg autorotates, so the filters see the displayed (rotated) frame.
pub fn video_filters(i: &VideoInfo, g: &Geometry, o: &PlanOpts, tonemap: bool) -> String {
    let mut f = Vec::new();
    if tonemap {
        f.push(TONEMAP.to_string());
    }
    f.push(format!("scale=w={}:h={}:flags=lanczos", g.width, g.height));
    if (i.sar - 1.0).abs() > f64::EPSILON {
        // Pixels are square after scaling to the display size.
        f.push("setsar=1".to_string());
    }
    if o.sharpen {
        f.push("unsharp=3:3:0.3:3:3:0.1".into());
    }
    f.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(w: u32, h: u32) -> VideoInfo {
        VideoInfo {
            codec: "h264".into(),
            width: w,
            height: h,
            sar: 1.0,
            rotation: 0,
            fps: 30.0,
            bitrate_kbps: 5000,
            duration: 10.0,
            pix_fmt: "yuv420p".into(),
            color_transfer: String::new(),
            has_audio: true,
            audio_channels: 2,
            video_index: 0,
            creation_time: None,
            tagged: false,
        }
    }

    fn dims(i: &VideoInfo) -> (u32, u32) {
        let g = geometry(i, &PlanOpts::default());
        (g.width, g.height)
    }

    #[test]
    fn landscape_1080p() {
        assert_eq!(dims(&info(1920, 1080)), (1280, 720));
    }

    #[test]
    fn never_upscales() {
        assert_eq!(dims(&info(640, 360)), (640, 360));
    }

    #[test]
    fn odd_sizes_become_even() {
        let (w, h) = dims(&info(641, 361));
        assert!(w % 2 == 0 && h % 2 == 0);
    }

    #[test]
    fn portrait_is_720_tall() {
        assert_eq!(dims(&info(1080, 1920)), (404, 720));
    }

    #[test]
    fn rotated_landscape_coded_is_portrait() {
        let mut i = info(1920, 1080);
        i.rotation = 90;
        assert_eq!(dims(&i), (404, 720));
    }

    #[test]
    fn anamorphic_sar_applied() {
        let mut i = info(720, 480);
        i.sar = 32.0 / 27.0;
        let (w, h) = dims(&i);
        assert!(w > h * 3 / 2 - 4 && h == 480);
    }

    #[test]
    fn filters_scale_then_setsar_then_sharpen() {
        let mut i = info(720, 480);
        i.sar = 32.0 / 27.0;
        let o = PlanOpts::default();
        let g = geometry(&i, &o);
        let f = video_filters(&i, &g, &o, false);
        assert!(
            f.starts_with("scale=w=") && f.contains(",setsar=1,unsharp="),
            "{f}"
        );
        let f = video_filters(&info(640, 360), &geometry(&info(640, 360), &o), &o, true);
        assert!(f.starts_with("zscale="));
    }

    #[test]
    fn fps_capped() {
        assert_eq!(geometry(&info(100, 100), &PlanOpts::default()).fps, 24.0);
        let mut i = info(100, 100);
        i.fps = 15.0;
        assert_eq!(geometry(&i, &PlanOpts::default()).fps, 15.0);
    }
}
