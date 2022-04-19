use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

use nannou::prelude::*;
use opencv::prelude::*;

use crate::texture;

pub struct ContourDetector {
    foreground_mask: Option<Mat>,
    request_sender: Sender<Mat>,
    response_receiver: Receiver<opencv::types::VectorOfMat>,
    pub texture: wgpu::Texture,
    worker_thread: thread::JoinHandle<()>,
    texture_uploader: texture::TextureUploader,
    width: i32,
    height: i32,
    finished: bool,
}

impl ContourDetector {
    pub fn new(app: &App, device: &wgpu::Device, size: Vec2) -> Self {
        let width = 650;
        let scale = width as f32 / size.x;
        let height = (size.y * scale).round() as i32;

        let texture = texture::create_texture(
            device,
            [width as u32, height as u32],
            wgpu::TextureFormat::Rgba16Float,
        );

        let (request_sender, request_receiver) = channel::<Mat>();
        let (response_sender, response_receiver) = channel::<opencv::types::VectorOfMat>();

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
                let mut f = unsafe {
                    opencv::core::Mat::new_rows_cols(width, height, opencv::core::CV_8UC1).unwrap()
                };

                let size = frame.size().unwrap();

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
            worker_thread,
            texture_uploader,
            width,
            height,
            finished: true,
        }
    }

    pub fn start_update(&mut self, frame: &Mat) {
        // let mut foreground_mask = unsafe {
        //     opencv::core::Mat::new_rows_cols(
        //         self.size.x as i32,
        //         self.size.y as i32,
        //         opencv::core::CV_8UC1,
        //     )
        //     .unwrap()
        // };

        self.finished = false;
        self.request_sender.send(frame.clone()).unwrap();

        // save result
        // self.foreground_mask = Some(foreground_mask);
    }

    pub fn finish_update(&mut self) {
        // let output_blobs = self.response_receiver.recv().unwrap();
        let output_blobs = match self.response_receiver.try_recv() {
            Ok(b) => b,
            Err(_) => return,
        };

        self.finished = true;
        // println!("detected {:?} objects", output_blobs.len());
    }

    pub fn is_finished(&self) -> bool { self.finished }

    pub fn update_texture(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let frame = match &self.foreground_mask {
            Some(s) => s,
            None => return,
        };

        texture::upload_mat_gray(device, encoder, frame, &self.texture, self.width as u32, self.height as u32);
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
