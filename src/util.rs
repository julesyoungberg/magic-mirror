use std::fs;
use std::path::PathBuf;

use nannou::prelude::*;

pub fn universal_path(input: String) -> String {
    PathBuf::from(input).into_os_string().into_string().unwrap()
}

pub fn create_texture(
    device: &wgpu::Device,
    size: [u32; 2],
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    wgpu::TextureBuilder::new()
        .size(size)
        .usage(wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING)
        .format(format)
        .build(device)
}

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
