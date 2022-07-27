use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

use mediapipe;
use nannou::prelude::*;
use opencv::prelude::*;

use crate::util;
use crate::faces;

pub struct HolisticDetector {
    detections: Vec<mediapipe::holistic::HolisticDetection>,
    request_sender: Sender<Mat>,
    response_receiver: Receiver<Vec<mediapipe::holistic::HolisticDetection>>,
    _worker_thread: thread::JoinHandle<()>,
}

impl HolisticDetector {
    pub fn new(video_size: Vec2) -> Self {
        let (request_sender, request_receiver) = channel::<Mat>();
        let (response_sender, response_receiver) = channel::<Vec<mediapipe::holistic::HolisticDetection>>();

        let worker_thread = thread::spawn(move || {
            let mut detector = mediapipe::holistic::HolisticDetector::default();

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

                let result = detector.process(&flip_frame);

                response_sender.send(vec![result]).unwrap();
            }
        });

        Self {
            detections: vec![],
            request_sender,
            response_receiver,
            _worker_thread: worker_thread,
        }
    }

    pub fn update(&mut self, frame: &Mat) {
        self.request_sender.send(frame.clone()).unwrap();

        match self.response_receiver.try_recv() {
            Ok(result) => {
                if !result.is_empty() {
                    self.detections = result;
                }
            }
            Err(_) => return,
        };
    }

    fn draw_detection(&self, draw: &Draw, detection: &mediapipe::holistic::HolisticDetection, mapper: &impl Fn(&Vec2) -> Vec2) {

        if let Some(face) = &detection.face {
            faces::draw_face(draw, &face.data.to_vec(), mapper);
        }

        if let Some(pose) = &detection.pose {
            // @todo
            println!("drawing pose");
        }

        if let Some(left_hand) = &detection.left_hand {
            // @todo
            println!("drawing left hand");
        }

        if let Some(right_hand) = &detection.right_hand {
            // @todo
            println!("drawing right hand");
        }
    }

    pub fn draw_detections(&self, draw: &Draw, _video_size: &Vec2, draw_size: &Vec2) {
        let hwidth = draw_size.x * 0.5;
        let hheight = draw_size.y * 0.5;

        let mapper = |input: &Vec2| {
            Vec2::new(
                util::map(input.x, 1.0, 0.0, hwidth, -hwidth) * 0.5,
                util::map(input.y, 0.0, 1.0, hheight, -hheight) * 0.5,
            )
        };

        for detection in &self.detections {
            self.draw_detection(draw, &detection, &mapper);
        }
    }
}