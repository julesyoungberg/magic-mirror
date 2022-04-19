use nannou::image;
use nannou::prelude::*;
use opencv::prelude::*;
use opencv::video;

use crate::util;

pub struct ContourDetector {
    background_subtractor: Box<dyn BackgroundSubtractor>,
    foreground_mask: Option<Mat>,
    model: opencv::dnn::Net,
    size: Vec2,
    pub texture: wgpu::Texture,
}

impl ContourDetector {
    pub fn new(app: &App, device: &wgpu::Device, size: Vec2) -> Self {
        let texture = util::create_texture(
            device,
            [size.x as u32, size.y as u32],
            wgpu::TextureFormat::Rgba16Float,
        );

        let weights_path = app.project_path().unwrap().join("Mask_R-CNN_weights");
        let model_path = weights_path
            .join("frozen_inference_graph_coco.pb")
            .into_os_string()
            .into_string()
            .unwrap();
        let config_path = weights_path
            .join("mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
            .into_os_string()
            .into_string()
            .unwrap();
        let model =
            opencv::dnn::read_net_from_tensorflow(model_path.as_str(), config_path.as_str())
                .unwrap();

        Self {
            background_subtractor: Box::new(
                video::create_background_subtractor_knn(10, 400.0, true).unwrap(),
            ),
            foreground_mask: None,
            model,
            size,
            texture,
        }
    }

    pub fn update(&mut self, frame: &Mat) {
        // let mut foreground_mask = unsafe {
        //     opencv::core::Mat::new_rows_cols(
        //         self.size.x as i32,
        //         self.size.y as i32,
        //         opencv::core::CV_8UC1,
        //     )
        //     .unwrap()
        // };

        // self.background_subtractor
        //     .apply(&frame, &mut foreground_mask, -1.0)
        //     .unwrap();

        let width = self.size.x as i32; // 650
        let height = self.size.y as i32; // 550

        let mut f = unsafe {
            opencv::core::Mat::new_rows_cols(width, height, opencv::core::CV_8UC1).unwrap()
        };

        let size = f.size().unwrap();

        opencv::imgproc::resize(&frame, &mut f, size, 0.0, 0.0, 0).unwrap();

        let blob = opencv::dnn::blob_from_image(
            &f,
            1.0,
            size,
            opencv::core::VecN::new(0.0, 0.0, 0.0, 0.0),
            false,
            false,
            opencv::core::CV_32F,
        )
        .unwrap();

        self.model
            .set_input(&blob, "", 1.0, opencv::core::VecN::new(0.0, 0.0, 0.0, 0.0))
            .unwrap();

        let mut output_blobs = opencv::types::VectorOfMat::new();

        let out_names = vec!["detection_out_final", "detection_masks"];
        self.model
            .forward(&mut output_blobs, &opencv::core::Vector::from(out_names))
            .unwrap();

        println!("detected {:?} objects", output_blobs.len());

        // save result
        // self.foreground_mask = Some(foreground_mask);
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
