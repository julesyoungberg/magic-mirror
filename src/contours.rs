use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

use nannou::prelude::*;
use opencv::prelude::*;

use crate::texture;

pub struct Object {
    class_id: i32,
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
    mask: Mat,
}

pub struct ContourDetector {
    foreground_mask: Option<Mat>,
    request_sender: Sender<Mat>,
    response_receiver: Receiver<Vec<Object>>,
    pub texture: wgpu::Texture,
    worker_thread: thread::JoinHandle<()>,
    texture_uploader: texture::TextureUploader,
    width: i32,
    height: i32,
    finished: bool,
}

// based on
// https://learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-rcnn-in-opencv-python-c/
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
        let (response_sender, response_receiver) = channel::<Vec<Object>>();

        let texture_uploader =
            texture::TextureUploader::new(texture::TextureType::Gray, width as u32, height as u32);

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
                // preprocess
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

                // run neural net
                model
                    .set_input(&blob, "", 1.0, opencv::core::VecN::new(0.0, 0.0, 0.0, 0.0))
                    .unwrap();

                let mut output_blobs = opencv::types::VectorOfMat::new();

                let out_names = vec!["detection_out_final", "detection_masks"];
                model
                    .forward(&mut output_blobs, &opencv::core::Vector::from(out_names))
                    .unwrap();

                // postprocess
                let mut out_detections = output_blobs.get(0).unwrap();
                let mut out_masks = output_blobs.get(1).unwrap();

                let detections_size = out_detections.mat_size();
                let num_detections = detections_size[2];
                // println!("num detections: {:?}", num_detections);

                let masks_size = out_masks.mat_size();
                let num_classes = masks_size[1];
                // println!("num classes: {:?}", num_classes);

                out_detections = out_detections
                    .reshape(1, out_detections.total() as i32 / 7)
                    .unwrap();

                let mut objects = vec![];

                for i in 0..num_detections {
                    let score: f32 = *out_detections.at_2d(i, 2).unwrap();

                    if score < 0.6 {
                        continue;
                    }

                    let class_id = *out_detections.at_2d::<f32>(i, 1).unwrap() as i32;
                    // println!("class id: {:?}", class_id);

                    let left = (width as f32 * *out_detections.at_2d::<f32>(i, 3).unwrap()) / scale;
                    let top = (height as f32 * *out_detections.at_2d::<f32>(i, 4).unwrap()) / scale;
                    let right =
                        (width as f32 * *out_detections.at_2d::<f32>(i, 5).unwrap()) / scale;
                    let bottom =
                        (height as f32 * *out_detections.at_2d::<f32>(i, 6).unwrap()) / scale;

                    // println!("left: {:?}, top: {:?}, right: {:?}, bottom: {:?}", left, top, right, bottom);

                    let mut mask_data = out_masks.ptr_2d_mut(i, class_id).unwrap();

                    let mut object_mask = unsafe {
                        Mat::new_rows_cols(
                            masks_size[2],
                            masks_size[3],
                            opencv::core::CV_32F,
                        )
                        .unwrap()
                    };

                    unsafe { object_mask.set_data(mask_data); };

                    // let mut countours_output = opencv::types::VectorOfMat::new();
                    
                    objects.push(Object {
                        class_id,
                        left,
                        top,
                        right,
                        bottom,
                        mask: object_mask,
                    });
                }

                response_sender.send(objects).unwrap();
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
        let detected_objects = match self.response_receiver.try_recv() {
            Ok(b) => b,
            Err(_) => return,
        };

        if !detected_objects.is_empty() {
            self.foreground_mask = Some(detected_objects[0].mask.clone())
        } else {
            self.foreground_mask = None;
        }

        self.finished = true;
        // println!("detected {:?} objects", output_blobs.len());
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn update_texture(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        let frame = match &self.foreground_mask {
            Some(s) => s,
            None => return,
        };

        texture::upload_mat_gray(
            device,
            encoder,
            frame,
            &self.texture,
            self.width as u32,
            self.height as u32,
        );
    }

    pub fn start_texture_upload(&self) {
        let frame = match &self.foreground_mask {
            Some(s) => s,
            None => return,
        };

        self.texture_uploader.start_upload(frame);
    }

    pub fn finish_texture_upload(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        self.texture_uploader
            .finish_upload(device, encoder, &self.texture);
    }
}
