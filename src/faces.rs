use dlib_face_recognition::*;
use nannou::prelude::*;
use opencv::prelude::*;

use crate::util;

#[derive(Clone, Debug)]
pub struct Face {
    pub position: Rectangle,
    pub landmarks: Vec<Vec2>,
}

impl Face {
    pub fn border_rect(&self) -> Vec<Vec2> {
        vec![
            Vec2::new(self.position.left as f32, self.position.top as f32),
            Vec2::new(self.position.right as f32, self.position.top as f32),
            Vec2::new(self.position.right as f32, self.position.bottom as f32),
            Vec2::new(self.position.left as f32, self.position.bottom as f32),
        ]
    }

    pub fn draw(&self, draw: &Draw, mapper: &impl Fn(&Vec2) -> Vec2) {
        let mut face_rect: Vec<Vec2> = self.border_rect().iter().map(mapper).collect();

        face_rect.push(face_rect[0].clone());

        draw.path()
            .stroke()
            .color(STEELBLUE)
            .weight(3.0)
            .points(face_rect);

        for landmark in &self.landmarks {
            let mapped = mapper(landmark);
            draw.ellipse()
                .color(STEELBLUE)
                .w(10.0)
                .h(10.0)
                .x_y(mapped.x, mapped.y);
        }
    }
}

pub struct FullFaceDetector {
    face_detector: FaceDetector,
    faces: Vec<Face>,
    landmark_predictor: LandmarkPredictor,
}

impl FullFaceDetector {
    pub fn new() -> Self {
        Self {
            face_detector: FaceDetector::default(),
            faces: vec![],
            landmark_predictor: LandmarkPredictor::default(),
        }
    }

    pub fn update(&mut self, frame: &Mat, video_size: Vec2) {
        let ptr = frame.datastart();
        let matrix = unsafe { ImageMatrix::new(video_size.x as usize, video_size.y as usize, ptr) };
        let face_locations = self.face_detector.face_locations(&matrix);

        self.faces = face_locations
            .iter()
            .map(|face| {
                let landmarks = self.landmark_predictor.face_landmarks(&matrix, &face);

                Face {
                    position: *face,
                    landmarks: landmarks
                        .iter()
                        .map(|l| Vec2::new(l.x() as f32, l.y() as f32))
                        .collect(),
                }
            })
            .collect();
    }

    pub fn draw_faces(&self, draw: &Draw, video_size: &Vec2, draw_size: &Vec2) {
        let hwidth = draw_size.x * 0.5;
        let hheight = draw_size.y * 0.5;

        let mapper = |input: &Vec2| {
            Vec2::new(
                util::map(input.x, 0.0, video_size.x, hwidth, -hwidth) * 0.5,
                util::map(input.y, 0.0, video_size.y, hheight, -hheight) * 0.5,
            )
        };

        for face in &self.faces {
            face.draw(draw, &mapper);
        }
    }
}

impl Default for FullFaceDetector {
    fn default() -> Self {
        Self::new()
    }
}