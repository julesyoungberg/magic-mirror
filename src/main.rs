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

#[derive(Clone, Debug)]
struct Face {
    position: Rectangle,
    landmarks: Vec<Vec2>,
}

impl Face {
    fn border_rect(&self) -> Vec<Vec2> {
        vec![
            Vec2::new(self.position.left as f32, self.position.top as f32),
            Vec2::new(self.position.right as f32, self.position.top as f32),
            Vec2::new(self.position.right as f32, self.position.bottom as f32),
            Vec2::new(self.position.left as f32, self.position.bottom as f32),
        ]
    }
}

struct Model {
    face_detector: FaceDetector,
    faces: Vec<Face>,
    landmark_predictor: LandmarkPredictor,
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
        face_detector: FaceDetector::default(),
        faces: vec![],
        landmark_predictor: LandmarkPredictor::default(),
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
        let ptr = frame.datastart();
        let matrix = unsafe {
            ImageMatrix::new(
                model.video_size.x as usize,
                model.video_size.y as usize,
                ptr,
            )
        };
        let face_locations = model.face_detector.face_locations(&matrix);

        model.faces = face_locations
            .iter()
            .map(|face| {
                let landmarks = model.landmark_predictor.face_landmarks(&matrix, &face);

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

fn map(input: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
    return (input - in_min) / (in_max - in_min) * (out_max - out_min) + out_min;
}

fn draw_faces(model: &Model, draw: &Draw) {
    let hwidth = model.size.x * 0.5;
    let hheight = model.size.y * 0.5;

    for face in &model.faces {
        let mut face_rect: Vec<Vec2> = face
            .border_rect()
            .iter()
            .map(|r| {
                Vec2::new(
                    map(r.x, 0.0, model.video_size.x, hwidth, -hwidth) * 0.5,
                    map(r.y, 0.0, model.video_size.y, hheight, -hheight) * 0.5,
                )
            })
            .collect();

        face_rect.push(face_rect[0].clone());

        draw.path()
            .stroke()
            .color(STEELBLUE)
            .weight(3.0)
            .points(face_rect);

        for landmark in &face.landmarks {
            draw.ellipse()
                .color(STEELBLUE)
                .w(10.0)
                .h(10.0)
                .x(map(landmark.x, 0.0, model.video_size.x, hwidth, -hwidth) * 0.5)
                .y(map(landmark.y, 0.0, model.video_size.y, hheight, -hheight) * 0.5);
        }
    }
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

    draw_faces(model, &draw);

    draw.to_frame(app, &frame).unwrap();
}
