use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

use mediapipe;
use nannou::prelude::*;
use opencv::prelude::*;

use crate::render::*;
use crate::texture;
use crate::util;

pub struct Segmentor {
    pub output_texture: wgpu::Texture,
    pub texture_reshaper: wgpu::TextureReshaper,
    video_width: u32,
    video_height: u32,
    request_sender: Sender<Mat>,
    response_receiver: Receiver<Mat>,
    worker_thread: thread::JoinHandle<()>,
}

impl Segmentor {
    pub fn new(device: &wgpu::Device, video_size: Vec2, sample_count: u32) -> Self {
        let (request_sender, request_receiver) = channel::<Mat>();
        let (response_sender, response_receiver) = channel::<Mat>();

        let worker_thread = thread::spawn(move || {
            let mut detector = mediapipe::segmentation::Segmentor::default();

            let mut rgb_frame = unsafe {
                opencv::core::Mat::new_rows_cols(
                    video_size.x.round() as i32,
                    video_size.y.round() as i32,
                    opencv::core::CV_8UC3,
                )
                .unwrap()
            };

            let mut flip_frame = unsafe {
                opencv::core::Mat::new_rows_cols(
                    video_size.x.round() as i32,
                    video_size.y.round() as i32,
                    opencv::core::CV_8UC3,
                )
                .unwrap()
            };

            for frame in request_receiver.iter() {
                opencv::imgproc::cvt_color(
                    &frame,
                    &mut rgb_frame,
                    opencv::imgproc::COLOR_BGR2RGB,
                    0,
                )
                .unwrap();

                opencv::core::flip(&rgb_frame, &mut flip_frame, 1).unwrap(); // horizontal

                let mut result = detector.process(&flip_frame);

                opencv::core::flip(&result, &mut flip_frame, 1).unwrap();

                opencv::imgproc::cvt_color(
                    &flip_frame,
                    &mut result,
                    opencv::imgproc::COLOR_RGB2BGR,
                    0,
                )
                .unwrap();

                response_sender.send(result).unwrap();
            }
        });

        let video_width = video_size.x as u32;
        let video_height = video_size.y as u32;

        let output_texture = create_app_texture(&device, video_width, video_height, 1);
        let texture_reshaper = create_texture_reshaper(&device, &output_texture, 1, sample_count);

        Self {
            output_texture,
            texture_reshaper,
            request_sender,
            response_receiver,
            worker_thread,
            video_width,
            video_height,
        }
    }

    pub fn update(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        frame: &Mat,
    ) {
        self.request_sender.send(frame.clone()).unwrap();

        match self.response_receiver.try_recv() {
            Ok(result) => {
                texture::upload_mat_rgb(
                    device,
                    encoder,
                    &result,
                    &self.output_texture,
                    self.video_width,
                    self.video_height,
                );
            }
            Err(_) => return,
        };
    }
}
