use std::fs;

use nannou::prelude::*;

/// Compiles a shader from the shaders directory
pub fn compile_shader(
    app: &App,
    device: &wgpu::Device,
    filename: &str,
    kind: shaderc::ShaderKind,
) -> wgpu::ShaderModule {
    let path = app
        .project_path()
        .unwrap()
        .join("src")
        .join("shaders")
        .join(filename)
        .into_os_string()
        .into_string()
        .unwrap();
    let code = fs::read_to_string(path).unwrap();
    let mut compiler = shaderc::Compiler::new().unwrap();
    let spirv = compiler
        .compile_into_spirv(code.as_str(), kind, filename, "main", None)
        .unwrap();
    wgpu::shader_from_spirv_bytes(device, spirv.as_binary_u8())
}

pub fn map(input: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
    return (input - in_min) / (in_max - in_min) * (out_max - out_min) + out_min;
}

pub fn float_as_bytes(data: &f32) -> [u8; 2] {
    half::f16::from_f32(*data).to_ne_bytes()
}

pub fn floats_as_byte_vec(data: &[f32]) -> Vec<u8> {
    let mut bytes = vec![];
    data.iter()
        .for_each(|f| bytes.extend(float_as_bytes(f).iter()));
    bytes
}

pub fn draw_landmarks<C: nannou::color::IntoLinSrgba<f32> + std::marker::Copy>(
    draw: &Draw, 
    landmarks: &Vec<mediapipe::Landmark>,
    color: C,
    mapper: &impl Fn(&Vec2) -> Vec2
) {
    for landmark in landmarks {
        let mapped = mapper(&Vec2::new(landmark.x, landmark.y));
        draw
            .ellipse()
            .color(color)
            .w(10.0)
            .h(10.0)
            .x_y(mapped.x, mapped.y);
    }
}
