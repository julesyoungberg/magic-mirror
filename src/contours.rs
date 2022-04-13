use nannou::image;
use nannou::prelude::*;
use opencv::prelude::*;

use crate::util;

pub struct ContourDetector {
    background: Option<Mat>,
    silhouette: Option<Mat>,
    size: Vec2,
    pub texture: wgpu::Texture,
}

impl ContourDetector {
    pub fn new(device: &wgpu::Device, size: Vec2) -> Self {
        let texture = util::create_texture(
            device,
            [size.x as u32, size.y as u32],
            wgpu::TextureFormat::Rgba16Float,
        );

        Self {
            background: None,
            silhouette: None,
            size,
            texture,
        }
    }

    pub fn set_background(&mut self, frame: Mat) {
        self.background = Some(frame);
    }

    pub fn update(&mut self, frame: &Mat) {
        let bg = match &self.background {
            Some(bg) => bg,
            None => return,
        };

        // subtract background
        let sub_result = match frame - bg {
            opencv::core::MatExprResult::Ok(result) => result,
            opencv::core::MatExprResult::Err(e) => {
                eprintln!("error: {:?}", e);
                self.silhouette = None;
                return;
            }
        };

        // get subtraction result
        let mut bg_diff = unsafe { Mat::from_raw(sub_result.into_raw()) };

        // convert to grayscale
        let mut gray_diff = Mat::default();
        opencv::imgproc::cvt_color(
            &mut bg_diff,
            &mut gray_diff,
            opencv::imgproc::ColorConversionCodes::COLOR_BGR2GRAY as i32,
            1,
        )
        .unwrap();

        // apply thresholding
        let mut silhouette = Mat::default();
        opencv::imgproc::threshold(
            &mut gray_diff,
            &mut silhouette,
            1.0,
            255.0,
            opencv::imgproc::ThresholdTypes::THRESH_BINARY as i32,
        )
        .unwrap();

        // @todo detect contours

        // save result
        self.silhouette = Some(silhouette);
    }

    pub fn update_texture(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let frame = match &self.silhouette {
            Some(s) => s,
            None => return,
        };

        let width = self.size.x as u32;
        let height = self.size.y as u32;

        util::upload_mat_to_texture(device, encoder, frame, &self.texture, width, height);
    }
}
