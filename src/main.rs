use nannou::prelude::*;

mod contours;
mod faces;
mod render;
mod texture;
mod uniforms;
mod util;
mod video_capture;
mod webcam;

use crate::contours::*;
use crate::faces::*;

fn main() {
    nannou::app(model).update(update).run();
}

struct Model {
    contour_detector: ContourDetector,
    contour_texture_reshaper: wgpu::TextureReshaper,
    face_detector: FullFaceDetector,
    size: Vec2,
    video_texture_reshaper: wgpu::TextureReshaper,
    video_size: Vec2,
    webcam_capture: webcam::WebcamCapture,
    first_run: bool,
}

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

fn model(app: &App) -> Model {
    // create window
    let main_window_id = app
        .new_window()
        .size(WIDTH, HEIGHT)
        .view(view)
        .build()
        .unwrap();

    let window = app.window(main_window_id).unwrap();
    let device = window.device();
    let sample_count = window.msaa_samples();

    println!("sample count: {:?}", sample_count);

    let (width, height) = window.inner_size_pixels();
    let size = pt2(width as f32, height as f32);

    let mut webcam_capture = webcam::WebcamCapture::new();

    webcam_capture.start_session(&device, size);

    let video_size = webcam_capture.video_capture.as_ref().unwrap().video_size;

    let video_texture = &webcam_capture.video_capture.as_ref().unwrap().video_texture;

    let video_texture_reshaper =
        render::create_texture_reshaper(&device, &video_texture, 1, sample_count);

    let contour_detector = ContourDetector::new(&app, &device, video_size);

    let contour_texture_reshaper =
        render::create_texture_reshaper(&device, &contour_detector.texture, 1, sample_count);

    Model {
        contour_detector,
        contour_texture_reshaper,
        face_detector: FullFaceDetector::new(video_size),
        size,
        video_texture_reshaper,
        video_size,
        webcam_capture,
        first_run: true,
    }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    let window = app.main_window();
    let device = window.device();

    model.webcam_capture.update();

    if let Some(frame) = model.webcam_capture.get_frame_ref() {
        // if !model.first_run {
        //     model.face_detector.finish_update();
        //     // model.contour_detector.finish_update();
        // } else {
        //     model.first_run = false;
        // }

        if model.face_detector.is_finished() {
            model.face_detector.start_update(frame);
        }

        if model.contour_detector.is_finished() {
            model.contour_detector.start_update(frame);
        }

        // The encoder we'll use to encode the compute pass and render pass.
        let desc = wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        };
        let mut encoder = device.create_command_encoder(&desc);

        model.contour_detector.update_texture(device, &mut encoder);
        // model.contour_detector.start_texture_upload();
        model.webcam_capture.update_texture(device, &mut encoder);
        // model.contour_detector.finish_texture_upload(device, &mut encoder);

        // submit encoded command buffer
        window.queue().submit(Some(encoder.finish()));

        // start processing frame
        // if model.face_detector.is_finished() && model.contour_detector.is_finished() {
        //     // model.contour_detector.start_update(frame);
        //     model.face_detector.start_update(frame);
        // }

        model.face_detector.finish_update();
        model.contour_detector.finish_update();
    }
}

fn view(app: &App, model: &Model, frame: Frame) {
    // Sample the texture and write it to the frame.
    {
        let mut encoder = frame.command_encoder();
        model
            // .video_texture_reshaper
            .contour_texture_reshaper
            .encode_render_pass(frame.texture_view(), &mut *encoder);
    }

    let draw = app.draw();

    model
        .face_detector
        .draw_faces(&draw, &model.video_size, &model.size);

    draw.to_frame(app, &frame).unwrap();
}
