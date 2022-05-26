use nannou::prelude::*;
use opencv::prelude::*;
use ringbuf::{Consumer, RingBuffer};
use std::fmt;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::SystemTime;
use std::{thread, time};

use crate::texture;

const FRAME_RATE: f64 = 30.0;

enum Message {
    Close(()),
    Pause(()),
    SetSpeed(f32),
    Unpause(()),
}

pub struct VideoConsumer {
    consumer: Consumer<opencv::core::Mat>,
}

impl fmt::Debug for VideoConsumer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "VideoConsumer")
    }
}

#[derive(Debug)]
pub struct VideoCapture {
    pub error: Option<String>,
    pub frame: Option<opencv::core::Mat>,
    pub running: bool,
    pub speed: f32,
    pub video_size: Vec2,
    pub video_texture: wgpu::Texture,

    capture_thread: Option<std::thread::JoinHandle<()>>,
    message_channel_tx: Sender<Message>,
    error_channel_rx: Receiver<String>,
    video_consumer: VideoConsumer,
    texture_uploader: texture::TextureUploader,
}

impl VideoCapture {
    pub fn new(
        device: &wgpu::Device,
        mut capture: opencv::videoio::VideoCapture,
        speed: f32,
    ) -> Self {
        // save size
        let width = capture.get(opencv::videoio::CAP_PROP_FRAME_WIDTH).unwrap();
        let height = capture.get(opencv::videoio::CAP_PROP_FRAME_HEIGHT).unwrap();
        let video_size = pt2(width as f32, height as f32);
        let mut frame_rate = FRAME_RATE;
        if let Ok(fr) = capture.get(opencv::videoio::CAP_PROP_FPS) {
            frame_rate = fr;
        }

        let texture_uploader =
            texture::TextureUploader::new(texture::TextureType::Rgb, width as u32, height as u32);

        // create video texture
        let video_texture = texture::create_texture(
            device,
            [width as u32, height as u32],
            wgpu::TextureFormat::Rgba16Float,
        );

        // setup ring buffer
        let video_ring_buffer = RingBuffer::<opencv::core::Mat>::new(2);
        let (mut video_producer, video_consumer) = video_ring_buffer.split();

        // setup communication channels
        let (error_channel_tx, error_channel_rx) = channel();
        let (message_channel_tx, message_channel_rx) = channel();

        // thread for reading from the capture
        let capture_thread = thread::spawn(move || {
            let clock = SystemTime::now();
            let mut video_speed = speed as f64;

            let mut frame = unsafe {
                opencv::core::Mat::new_rows_cols(
                    height.round() as i32,
                    width.round() as i32,
                    opencv::core::CV_8UC3,
                )
                .unwrap()
            };

            'capture: loop {
                // read from camera
                let start_time = clock.elapsed().unwrap().as_secs_f64();
                match capture.read(&mut frame) {
                    Ok(success) => {
                        if !success {
                            println!("No video frame available");
                            // capture
                            //     .set(opencv::videoio::CAP_PROP_POS_FRAMES, 0.0)
                            //     .unwrap();
                            continue 'capture;
                        }
                    }
                    Err(e) => {
                        println!("Error capturing video frame: {:?}", e);
                        error_channel_tx.send(e.to_string()).unwrap();
                        break 'capture;
                    }
                }

                video_producer.push(frame.clone()).ok();

                if let Ok(msg) = message_channel_rx.try_recv() {
                    match msg {
                        Message::Close(()) => {
                            // break from the outer loop
                            println!("Closing capture thread");
                            break 'capture;
                        }
                        Message::Pause(()) => {
                            // the stream has been paused, block it is unpaused
                            'pause: for message in message_channel_rx.iter() {
                                match message {
                                    Message::Close(()) => break 'capture,
                                    Message::SetSpeed(s) => video_speed = s as f64,
                                    Message::Unpause(()) => break 'pause,
                                    _ => (),
                                }
                            }
                        }
                        Message::SetSpeed(s) => video_speed = s as f64,
                        Message::Unpause(()) => (),
                    }
                }

                let frame_dur = 1.0_f64 / (frame_rate * video_speed);
                let elapsed = clock.elapsed().unwrap().as_secs_f64() - start_time;
                let extra_time = frame_dur - elapsed;
                if extra_time > 0.01 {
                    thread::sleep(time::Duration::from_millis(
                        ((extra_time - 0.01) * 1000.0) as u64,
                    ));
                }
            }
        });

        Self {
            capture_thread: Some(capture_thread),
            message_channel_tx,
            error: None,
            error_channel_rx,
            frame: None,
            running: true,
            speed,
            video_consumer: VideoConsumer {
                consumer: video_consumer,
            },
            video_size,
            video_texture,
            texture_uploader,
        }
    }

    pub fn end_session(&mut self) {
        if !self.running {
            return;
        }

        self.message_channel_tx.send(Message::Close(())).ok();
        if let Some(handle) = self.capture_thread.take() {
            handle.join().ok();
        }

        self.running = false;
    }

    pub fn update(&mut self) {
        if !self.running {
            return;
        }

        // check the error channel for errors
        if let Ok(err) = self.error_channel_rx.try_recv() {
            println!("Webcam error: {:?}", err);
            self.error = Some(err);
            self.end_session();
            return;
        }

        self.frame = self.video_consumer.consumer.pop();
    }

    pub fn update_texture(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        // if !self.running {
        //     return;
        // }

        let frame = match &self.frame {
            Some(f) => f,
            None => return,
        };

        let width = self.video_size.x as u32;
        let height = self.video_size.y as u32;

        println!("uploading texture");
        texture::upload_mat_rgb(device, encoder, frame, &self.video_texture, width, height);
    }

    pub fn start_texture_upload(&self) {
        let frame = match &self.frame {
            Some(f) => f,
            None => return,
        };

        self.texture_uploader.start_upload(frame);
    }

    pub fn finish_texture_upload(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        self.texture_uploader
            .finish_upload(device, encoder, &self.video_texture);
    }

    // pub fn pause(&mut self) {
    //     self.message_channel_tx.send(Message::Pause(())).ok();
    // }

    // pub fn unpause(&mut self) {
    //     self.message_channel_tx.send(Message::Unpause(())).ok();
    // }

    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed;
        self.message_channel_tx.send(Message::SetSpeed(speed)).ok();
    }
}
