// use nannou::math::cgmath::Matrix4;
use nannou::prelude::*;

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
