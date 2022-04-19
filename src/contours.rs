use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

use nannou::prelude::*;
use opencv::prelude::*;

use crate::texture;

pub struct ContourDetector {
    foreground_mask: Option<Mat>,
    request_sender: Sender<Mat>,
    response_receiver: Receiver<opencv::types::VectorOfMat>,
    size: Vec2,
    pub texture: wgpu::Texture,
    worker_thread: std::thread::JoinHandle<()>,
    texture_uploader: texture::TextureUploader,
}

impl ContourDetector {
    pub fn new(app: &App, device: &wgpu::Device, size: Vec2) -> Self {
        let texture = texture::create_texture(
            device,
            [size.x as u32, size.y as u32],
            wgpu::TextureFormat::Rgba16Float,
        );

        let (request_sender, request_receiver) = channel::<Mat>();
        let (response_sender, response_receiver) = channel::<opencv::types::VectorOfMat>();

        let width = size.x as i32; // 650
        let height = size.y as i32; // 550

        let texture_uploader = texture::TextureUploader::new(texture::TextureType::Gray, width as u32, height as u32);

        let project_path = app.project_path();

        let worker_thread = thread::spawn(move || {
            let weights_path = project_path.unwrap().join("Mask_R-CNN_weights");
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
            let mut model =
                opencv::dnn::read_net_from_tensorflow(model_path.as_str(), config_path.as_str())
                    .unwrap();

            for frame in request_receiver.iter() {
                // let mut f = unsafe {
                //     opencv::core::Mat::new_rows_cols(width, height, opencv::core::CV_8UC1).unwrap()
                // };

                let size = frame.size().unwrap();

                // opencv::imgproc::resize(&frame, &mut f, size, 0.0, 0.0, 0).unwrap();

                let blob = opencv::dnn::blob_from_image(
                    &frame,
                    1.0,
                    size,
                    opencv::core::VecN::new(0.0, 0.0, 0.0, 0.0),
                    false,
                    false,
                    opencv::core::CV_32F,
                )
                .unwrap();

                model
                    .set_input(&blob, "", 1.0, opencv::core::VecN::new(0.0, 0.0, 0.0, 0.0))
                    .unwrap();

                let mut output_blobs = opencv::types::VectorOfMat::new();

                let out_names = vec!["detection_out_final", "detection_masks"];
                model
                    .forward(&mut output_blobs, &opencv::core::Vector::from(out_names))
                    .unwrap();

                response_sender.send(output_blobs).unwrap();
            }
        });

        Self {
            foreground_mask: None,
            texture,
            request_sender,
            response_receiver,
            size,
            worker_thread,
            texture_uploader,
        }
    }

    pub fn start_update(&self, frame: &Mat) {
        // let mut foreground_mask = unsafe {
        //     opencv::core::Mat::new_rows_cols(
        //         self.size.x as i32,
        //         self.size.y as i32,
        //         opencv::core::CV_8UC1,
        //     )
        //     .unwrap()
        // };

        self.request_sender.send(frame.clone()).unwrap();

        // save result
        // self.foreground_mask = Some(foreground_mask);
    }

    pub fn finish_update(&mut self) {
        let output_blobs = self.response_receiver.recv().unwrap();
        // println!("detected {:?} objects", output_blobs.len());
    }

    pub fn update_texture(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let frame = match &self.foreground_mask {
            Some(s) => s,
            None => return,
        };

        let width = self.size.x as u32;
        let height = self.size.y as u32;

        texture::upload_mat_gray(device, encoder, frame, &self.texture, width, height);
    }

    pub fn start_texture_upload(&self) {
        let frame = match &self.foreground_mask {
            Some(s) => s,
            None => return,
        };

        self.texture_uploader.start_upload(frame);
    }

    pub fn finish_texture_upload(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        self.texture_uploader.finish_upload(device, encoder, &self.texture);
    }
}
