use dlib_face_recognition::*;
use nannou::prelude::*;
use opencv::prelude::*;

mod render;
mod uniforms;
mod util;
mod video_capture;
mod webcam;

fn main() {
    nannou::app(model).update(update).run();
}

struct Model {
    face_detector: FaceDetector,
    landmark_predictor: LandmarkPredictor,
    render: render::CustomRenderer,
    size: Vec2,
    uniforms: uniforms::UniformBuffer,
    webcam_capture: webcam::WebcamCapture,
}

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;

fn model(app: &App) -> Model {
    // create window
    let main_window_id = app
        .new_window()
        .size(1920, 1080)
        .view(view)
        .build()
        .unwrap();

    let window = app.window(main_window_id).unwrap();
    let device = window.device();
    let sample_count = window.msaa_samples();

    println!("sample count: {:?}", sample_count);

    let vs_mod = util::compile_shader(app, device, "default.vert", shaderc::ShaderKind::Vertex);
    let fs_mod = util::compile_shader(app, device, "default.frag", shaderc::ShaderKind::Fragment);

    // Create the buffer that will store the uniforms.
    let uniforms = uniforms::UniformBuffer::new(device, WIDTH as f32, HEIGHT as f32);

    let (width, height) = window.inner_size_pixels();
    let size = pt2(width as f32, height as f32);

    let mut webcam_capture = webcam::WebcamCapture::new();

    webcam_capture.start_session(&device, size);

    let textures = Some(vec![
        &webcam_capture.video_capture.as_ref().unwrap().video_texture,
    ]);

    let sampler = wgpu::SamplerBuilder::new().build(device);

    let render = render::CustomRenderer::new::<uniforms::Uniforms>(
        device,
        &vs_mod,
        &fs_mod,
        None,
        None,
        textures.as_ref(),
        Some(&sampler),
        Some(&uniforms.buffer),
        WIDTH,
        HEIGHT,
        1,
        sample_count,
    )
    .unwrap();

    Model {
        face_detector: FaceDetector::default(),
        landmark_predictor: LandmarkPredictor::default(),
        render,
        size,
        uniforms,
        webcam_capture,
    }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    let window = app.main_window();
    let device = window.device();

    model.webcam_capture.update();

    let frame_ref = model
        .webcam_capture
        .video_capture
        .as_ref()
        .unwrap()
        .frame
        .as_ref();

    if let Some(frame) = frame_ref {
        let video_size = model
            .webcam_capture
            .video_capture
            .as_ref()
            .unwrap()
            .video_size;
        let ptr = frame.datastart();
        let matrix = unsafe { ImageMatrix::new(video_size.x as usize, video_size.y as usize, ptr) };
        let face_locations = model.face_detector.face_locations(&matrix);
        println!("num faces: {:?}", face_locations.len());

        for face in face_locations.iter() {
            println!("face location: {:?}", face);

            let landmarks = model.landmark_predictor.face_landmarks(&matrix, &face);

            for landmark in landmarks.iter() {
                println!("landmark: {:?}", landmark);
            }
        }
    }

    // The encoder we'll use to encode the compute pass and render pass.
    let desc = wgpu::CommandEncoderDescriptor {
        label: Some("encoder"),
    };
    let mut encoder = device.create_command_encoder(&desc);

    model.webcam_capture.update_texture(device, &mut encoder);

    model.uniforms.update(device, &mut encoder);

    model.render.render(&mut encoder);

    // submit encoded command buffer
    window.queue().submit(Some(encoder.finish()));
}

fn view(_app: &App, model: &Model, frame: Frame) {
    // Sample the texture and write it to the frame.
    let mut encoder = frame.command_encoder();
    model
        .render
        .texture_reshaper
        .encode_render_pass(frame.texture_view(), &mut *encoder);
}
