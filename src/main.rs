use nannou::prelude::*;

mod faces;
mod render;
mod uniforms;
mod util;
mod video_capture;
mod webcam;

use crate::faces::*;

fn main() {
    nannou::app(model).update(update).run();
}

struct Model {
    face_detector: FullFaceDetector,
    render: render::CustomRenderer,
    size: Vec2,
    video_size: Vec2,
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

    let video_size = webcam_capture.video_capture.as_ref().unwrap().video_size;

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
        face_detector: FullFaceDetector::default(),
        render,
        size,
        video_size,
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
        model.face_detector.update(frame, model.video_size);
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

fn view(app: &App, model: &Model, frame: Frame) {
    // Sample the texture and write it to the frame.
    {
        let mut encoder = frame.command_encoder();
        model
            .render
            .texture_reshaper
            .encode_render_pass(frame.texture_view(), &mut *encoder);
    }

    let draw = app.draw();

    model
        .face_detector
        .draw_faces(&draw, &model.video_size, &model.size);

    draw.to_frame(app, &frame).unwrap();
}
