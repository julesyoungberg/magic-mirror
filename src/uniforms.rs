use nannou::prelude::*;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct Uniforms {
    pub width: f32,
    pub height: f32,
}

impl Uniforms {
    pub fn new(width: f32, height: f32) -> Self {
        Uniforms { width, height }
    }
}

fn as_bytes(uniforms: &Uniforms) -> &[u8] {
    unsafe { wgpu::bytes::from(uniforms) }
}

pub struct UniformBuffer {
    pub data: Uniforms,
    pub buffer: wgpu::Buffer,
}

impl UniformBuffer {
    pub fn new(device: &wgpu::Device, width: f32, height: f32) -> Self {
        let data = Uniforms::new(width, height);

        let uniforms_bytes = as_bytes(&data);
        let usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        let buffer = device.create_buffer_init(&wgpu::BufferInitDescriptor {
            label: Some("uniform-buffer"),
            contents: uniforms_bytes,
            usage,
        });

        Self { data, buffer }
    }

    pub fn update(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        // An update for the uniform buffer with the current time.
        let uniforms_bytes = as_bytes(&self.data);
        let uniforms_size = uniforms_bytes.len();
        let usage = wgpu::BufferUsages::COPY_SRC;
        let new_uniform_buffer = device.create_buffer_init(&wgpu::BufferInitDescriptor {
            label: None,
            contents: uniforms_bytes,
            usage,
        });

        encoder.copy_buffer_to_buffer(
            &new_uniform_buffer,
            0,
            &self.buffer,
            0,
            uniforms_size as u64,
        );
    }
}
