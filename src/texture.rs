use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

use nannou::image;
use nannou::prelude::*;
use opencv::prelude::*;

use crate::util::*;

pub enum TextureType {
    Rgb,
    Gray,
}

#[derive(Debug)]
pub struct TextureUploader {
    request_sender: Sender<Mat>,
    response_receiver: Receiver<Vec<u8>>,
    worker_thread: std::thread::JoinHandle<()>,
}

impl TextureUploader {
    pub fn new(texture_type: TextureType, width: u32, height: u32) -> Self {
        let (request_sender, request_receiver) = channel::<Mat>();
        let (response_sender, response_receiver) = channel::<Vec<u8>>();

        let worker_thread = thread::spawn(move || {
            for frame in request_receiver.iter() {
                let byte_vec = match texture_type {
                    TextureType::Rgb => {
                        let frame_data: Vec<Vec<opencv::core::Vec3b>> = frame.to_vec_2d().unwrap();

                        let image = image::ImageBuffer::from_fn(width, height, |x, y| {
                            let pixel = frame_data[y as usize][(width - x - 1) as usize];
                            // convert from BGR to RGB
                            image::Rgba([
                                pixel[2] as f32 / 255.0,
                                pixel[1] as f32 / 255.0,
                                pixel[0] as f32 / 255.0,
                                1.0,
                            ])
                        });

                        let flat_samples = image.as_flat_samples();
                        floats_as_byte_vec(flat_samples.as_slice())
                    }
                    TextureType::Gray => {
                        let frame_data: &[u8] = frame.data_bytes().unwrap();

                        let image = image::ImageBuffer::from_fn(width, height, |x, y| {
                            let index = (y * width + (width - x - 1)) as usize;
                            let luma = frame_data[index] as f32 / 255.0;
                            image::Rgba([luma, luma, luma, 1.0])
                        });

                        let flat_samples = image.as_flat_samples();
                        floats_as_byte_vec(flat_samples.as_slice())
                    }
                };

                response_sender.send(byte_vec).unwrap();
            }
        });

        Self {
            request_sender,
            response_receiver,
            worker_thread,
        }
    }

    pub fn start_upload(&self, frame: &Mat) {
        self.request_sender.send(frame.clone()).unwrap();
    }

    pub fn finish_upload(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture) {
        let bytes = self.response_receiver.recv().unwrap();
        texture.upload_data(device, encoder, &bytes);
    }
}

pub fn create_texture(
    device: &wgpu::Device,
    size: [u32; 2],
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    wgpu::TextureBuilder::new()
        .size(size)
        .usage(wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING)
        .format(format)
        .build(device)
}

pub fn upload_mat_rgb(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    frame: &Mat,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) {
    let frame_data: Vec<Vec<opencv::core::Vec3b>> = frame.to_vec_2d().unwrap();

    let image = image::ImageBuffer::from_fn(width, height, |x, y| {
        let pixel = frame_data[y as usize][(width - x - 1) as usize];
        // convert from BGR to RGB
        image::Rgba([
            pixel[2] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[0] as f32 / 255.0,
            1.0,
        ])
    });

    let flat_samples = image.as_flat_samples();
    let byte_vec = floats_as_byte_vec(flat_samples.as_slice());

    texture.upload_data(device, encoder, &byte_vec);
}

pub fn upload_mat_gray(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    frame: &Mat,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) {
    let frame_data: &[u8] = frame.data_bytes().unwrap();

    let image = image::ImageBuffer::from_fn(width, height, |x, y| {
        let index = (y * width + (width - x - 1)) as usize;
        let luma = frame_data[index] as f32 / 255.0;
        image::Rgba([luma, luma, luma, 1.0])
    });

    let flat_samples = image.as_flat_samples();
    let byte_vec = floats_as_byte_vec(flat_samples.as_slice());

    texture.upload_data(device, encoder, &byte_vec);
}
