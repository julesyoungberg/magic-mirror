use nannou::image;
use nannou::prelude::*;
use opencv::prelude::*;
use opencv::video;

use crate::util;

pub struct ContourDetector {
    background_subtractor: Box<dyn BackgroundSubtractor>,
    foreground_mask: Option<Mat>,
    size: Vec2,
    pub texture: wgpu::Texture,
}

impl ContourDetector {
    pub fn new(device: &wgpu::Device, size: Vec2) -> Self {
        let texture = util::create_texture(
            device,
            [size.x as u32, size.y as u32],
            wgpu::TextureFormat::R16Float,
        );

        Self {
            background_subtractor: Box::new(
                video::create_background_subtractor_knn(500, 400.0, false).unwrap(),
            ),
            foreground_mask: None,
            size,
            texture,
        }
    }

    pub fn update(&mut self, frame: &Mat) {
        let mut foreground_mask = unsafe {
            opencv::core::Mat::new_rows_cols(
                self.size.x as i32,
                self.size.y as i32,
                opencv::core::CV_8UC1,
            )
            .unwrap()
        };

        self.background_subtractor
            .apply(&frame, &mut foreground_mask, -1.0)
            .unwrap();

        // @todo detect contours

        // save result
        self.foreground_mask = Some(foreground_mask);
    }

    pub fn update_texture(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let frame = match &self.foreground_mask {
            Some(s) => s,
            None => return,
        };

        let width = self.size.x as u32;
        let height = self.size.y as u32;

        util::upload_mat_to_texture_gray(device, encoder, frame, &self.texture, width, height);
    }
}
