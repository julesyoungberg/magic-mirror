use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;

use mediapipe;
use nannou::prelude::*;
use opencv::prelude::*;

use crate::util;

pub struct FullFaceDetector {
    faces: Vec<mediapipe::FaceMesh>,
    request_sender: Sender<Mat>,
    response_receiver: Receiver<Option<mediapipe::FaceMesh>>,
    worker_thread: thread::JoinHandle<()>,
    finished: bool,
}

impl FullFaceDetector {
    pub fn new(video_size: Vec2) -> Self {
        let (request_sender, request_receiver) = channel::<Mat>();
        let (response_sender, response_receiver) = channel::<Option<mediapipe::FaceMesh>>();

        let worker_thread = thread::spawn(move || {
            let mut detector = mediapipe::face_mesh::FaceMeshDetector::default();

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

                response_sender.send(result).unwrap();
            }
        });

        Self {
            faces: vec![],
            request_sender,
            response_receiver,
            worker_thread,
            finished: true,
        }
    }

    pub fn start_update(&mut self, frame: &Mat) {
        self.finished = false;
        self.request_sender.send(frame.clone()).unwrap();
    }

    pub fn finish_update(&mut self) {
        match self.response_receiver.try_recv() {
            Ok(result) => {
                if let Some(mesh) = result {
                    self.faces = vec![mesh];
                }
            }
            Err(_) => return,
        };

        self.finished = true;
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn draw_face(
        &self,
        draw: &Draw,
        face: &mediapipe::FaceMesh,
        mapper: &impl Fn(&Vec2) -> Vec2,
    ) {
        for landmark in &face.data {
            let mapped = mapper(&Vec2::new(landmark.x, landmark.y));
            draw.ellipse()
                .color(STEELBLUE)
                .w(10.0)
                .h(10.0)
                .x_y(mapped.x, mapped.y);
        }
    }

    pub fn draw_faces(&self, draw: &Draw, video_size: &Vec2, draw_size: &Vec2) {
        let hwidth = draw_size.x * 0.5;
        let hheight = draw_size.y * 0.5;

        let mapper = |input: &Vec2| {
            Vec2::new(
                util::map(input.x, 1.0, 0.0, hwidth, -hwidth) * 0.5,
                util::map(input.y, 0.0, 1.0, hheight, -hheight) * 0.5,
            )
        };

        for face in &self.faces {
            self.draw_face(draw, &face, &mapper);
        }
    }
}
