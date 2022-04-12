use nannou::prelude::*;

mod bufferable;
mod util;
mod video_capture;
mod webcam;

fn main() {
    nannou::app(model).update(update).run();
}

struct Model {
    webcam_capture: webcam::WebcamCapture,
}

fn model(app: &App) -> Model {
    // create window
    let main_window_id = app
        .new_window()
        .size(1920, 1080)
        .view(view)
        .build()
        .unwrap();

    let window = app.window(main_window_id).unwrap();

    let (width, height) = window.inner_size_pixels();
    let size = pt2(width as f32, height as f32);

    let device = window.device();

    let mut webcam_capture = webcam::WebcamCapture::new();

    webcam_capture.start_session(&device, size);

    Model { webcam_capture }
}

fn update(_app: &App, model: &mut Model, _update: Update) {
    model.webcam_capture.update();
}

fn view(_app: &App, _model: &Model, frame: Frame) {
    frame.clear(PURPLE);
}
