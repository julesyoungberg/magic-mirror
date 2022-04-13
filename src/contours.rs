use nannou::prelude::*;
use opencv::prelude::*;

pub struct ContourDetector {
    background: Option<Mat>,
    silhouette: Option<Mat>,
}

impl ContourDetector {
    pub fn new() -> Self {
        Self {
            background: None,
            silhouette: None,
        }
    }

    pub fn set_background(&mut self, frame: Mat) {
        self.background = Some(frame);
    }

    pub fn update(&mut self, frame: &Mat, video_size: Vec2) {
        let bg = match self.background.as_ref() {
            Some(bg) => bg,
            None => return,
        };

        // subtract background
        let sub_result = match frame - bg {
            opencv::core::MatExprResult::Ok(result) => result,
            opencv::core::MatExprResult::Err(e) => {
                eprintln!("error: {:?}", e);
                self.silhouette = None;
                return;
            }
        };

        // get subtraction result
        let mut bg_diff = unsafe { Mat::from_raw(sub_result.into_raw()) };

        // convert to grayscale
        let mut gray_diff = Mat::default();
        opencv::imgproc::cvt_color(
            &mut bg_diff,
            &mut gray_diff,
            opencv::imgproc::ColorConversionCodes::COLOR_BGR2GRAY as i32,
            1,
        )
        .unwrap();

        // apply thresholding
        let mut silhouette = Mat::default();
        opencv::imgproc::threshold(
            &mut gray_diff,
            &mut silhouette,
            1.0,
            255.0,
            opencv::imgproc::ThresholdTypes::THRESH_BINARY as i32,
        )
        .unwrap();

        // @todo detect contours

        // save result
        self.silhouette = Some(silhouette);
    }
}

impl Default for ContourDetector {
    fn default() -> Self {
        Self::new()
    }
}
