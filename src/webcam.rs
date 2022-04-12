use nannou::prelude::*;
use opencv::prelude::*;

use crate::bufferable::Bufferable;
use crate::video_capture::VideoCapture;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Data {
    pub video_size: Vec2,
}

pub struct WebcamCapture {
    pub updated: bool,
    pub video_capture: Option<VideoCapture>,

    data: Data,
}

impl Bufferable<Data> for WebcamCapture {
    fn as_bytes(&self) -> &[u8] {
        unsafe { wgpu::bytes::from(&self.data) }
    }

    fn textures(&self) -> Vec<&wgpu::Texture> {
        match &self.video_capture {
            Some(capture) => vec![&capture.video_texture],
            None => vec![],
        }
    }
}

impl WebcamCapture {
    pub fn new() -> Self {
        Self {
            data: Data {
                video_size: pt2(0.0, 0.0),
            },
            updated: false,
            video_capture: None,
        }
    }

    /// Starts a webcam session.
    /// Spawns a thread to consumer webcam data with OpenCV.
    pub fn start_session(&mut self, device: &wgpu::Device, size: Point2) {
        if let Some(video_capture) = &self.video_capture {
            if video_capture.running {
                return;
            }
        }

        let mut capture = opencv::videoio::VideoCapture::new(0, opencv::videoio::CAP_ANY).unwrap();
        capture
            .set(opencv::videoio::CAP_PROP_FRAME_WIDTH, size[0] as f64)
            .ok();
        capture
            .set(opencv::videoio::CAP_PROP_FRAME_HEIGHT, size[1] as f64)
            .ok();

        self.video_capture = Some(VideoCapture::new(device, capture, 1.0));

        self.updated = true;
    }

    pub fn end_session(&mut self) {
        if let Some(video_capture) = &mut self.video_capture {
            video_capture.end_session();
            self.video_capture = None;
        }
    }

    pub fn update(&mut self) {
        if let Some(video_capture) = &mut self.video_capture {
            video_capture.update();
            self.data.video_size = video_capture.video_size;
        }
    }

    pub fn update_texture(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        if let Some(video_capture) = &self.video_capture {
            video_capture.update_texture(device, encoder);
        }
    }

    pub fn pause(&mut self) {
        if let Some(video_capture) = &mut self.video_capture {
            video_capture.pause();
        }
    }

    pub fn unpause(&mut self) {
        if let Some(video_capture) = &mut self.video_capture {
            video_capture.unpause();
        }
    }
}
