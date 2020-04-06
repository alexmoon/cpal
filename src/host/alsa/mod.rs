extern crate alsa;
extern crate libc;

use crate::{
    BackendSpecificError, BuildStreamError, ChannelCount, Data, DefaultStreamConfigError,
    DeviceNameError, DevicesError, PauseStreamError, PlayStreamError, SampleFormat, SampleRate,
    StreamConfig, StreamError, SupportedStreamConfig, SupportedStreamConfigRange,
    SupportedStreamConfigsError,
};
use std::cmp;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::vec::IntoIter as VecIntoIter;
use traits::{DeviceTrait, HostTrait, StreamTrait};

pub use self::enumerate::{default_input_device, default_output_device, Devices};

pub type SupportedInputConfigs = VecIntoIter<SupportedStreamConfigRange>;
pub type SupportedOutputConfigs = VecIntoIter<SupportedStreamConfigRange>;

mod enumerate;

/// The default linux, dragonfly and freebsd host type.
#[derive(Debug)]
pub struct Host;

impl Host {
    pub fn new() -> Result<Self, crate::HostUnavailable> {
        Ok(Host)
    }
}

impl HostTrait for Host {
    type Devices = Devices;
    type Device = Device;

    fn is_available() -> bool {
        // Assume ALSA is always available on linux/dragonfly/freebsd.
        true
    }

    fn devices(&self) -> Result<Self::Devices, DevicesError> {
        Devices::new()
    }

    fn default_input_device(&self) -> Option<Self::Device> {
        default_input_device()
    }

    fn default_output_device(&self) -> Option<Self::Device> {
        default_output_device()
    }
}

impl DeviceTrait for Device {
    type SupportedInputConfigs = SupportedInputConfigs;
    type SupportedOutputConfigs = SupportedOutputConfigs;
    type Stream = Stream;

    fn name(&self) -> Result<String, DeviceNameError> {
        Device::name(self)
    }

    fn supported_input_configs(
        &self,
    ) -> Result<Self::SupportedInputConfigs, SupportedStreamConfigsError> {
        Device::supported_input_configs(self)
    }

    fn supported_output_configs(
        &self,
    ) -> Result<Self::SupportedOutputConfigs, SupportedStreamConfigsError> {
        Device::supported_output_configs(self)
    }

    fn default_input_config(&self) -> Result<SupportedStreamConfig, DefaultStreamConfigError> {
        Device::default_input_config(self)
    }

    fn default_output_config(&self) -> Result<SupportedStreamConfig, DefaultStreamConfigError> {
        Device::default_output_config(self)
    }

    fn build_input_stream_raw<D, E>(
        &self,
        conf: &StreamConfig,
        sample_format: SampleFormat,
        data_callback: D,
        error_callback: E,
    ) -> Result<Self::Stream, BuildStreamError>
    where
        D: FnMut(&Data) + Send + 'static,
        E: FnMut(StreamError) + Send + 'static,
    {
        let stream_inner =
            self.build_stream_inner(conf, sample_format, alsa::Direction::Capture)?;
        let stream = Stream::new_input(Arc::new(stream_inner), data_callback, error_callback);
        Ok(stream)
    }

    fn build_output_stream_raw<D, E>(
        &self,
        conf: &StreamConfig,
        sample_format: SampleFormat,
        data_callback: D,
        error_callback: E,
    ) -> Result<Self::Stream, BuildStreamError>
    where
        D: FnMut(&mut Data) + Send + 'static,
        E: FnMut(StreamError) + Send + 'static,
    {
        let stream_inner =
            self.build_stream_inner(conf, sample_format, alsa::Direction::Playback)?;
        let stream = Stream::new_output(Arc::new(stream_inner), data_callback, error_callback);
        Ok(stream)
    }
}

pub struct Device {
    name: String,
    playback: Mutex<Option<alsa::pcm::PCM>>,
    capture: Mutex<Option<alsa::pcm::PCM>>,
}

impl Device {
    fn build_stream_inner(
        &self,
        conf: &StreamConfig,
        sample_format: SampleFormat,
        stream_type: alsa::Direction,
    ) -> Result<StreamInner, BuildStreamError> {
        let handle = {
            let mut guard = match stream_type {
                alsa::Direction::Playback => self.playback.lock(),
                alsa::Direction::Capture => self.capture.lock(),
            }
            .unwrap();
            guard.take()
        };

        let handle = match handle {
            Some(h) => h,
            None => alsa::pcm::PCM::new(&self.name, stream_type, true)?,
        };
        set_hw_params_from_format(&handle, conf, sample_format)?;
        let period_len = set_sw_params_from_format(&handle, conf, stream_type)?;

        let stream_inner = StreamInner {
            channel: handle,
            sample_format,
            num_channels: conf.channels as u16,
            period_len,
            state: (Mutex::new(StreamState::Paused), Condvar::new()),
        };

        Ok(stream_inner)
    }

    #[inline]
    fn name(&self) -> Result<String, DeviceNameError> {
        Ok(self.name.clone())
    }

    fn supported_configs(
        &self,
        stream_t: alsa::Direction,
    ) -> Result<VecIntoIter<SupportedStreamConfigRange>, SupportedStreamConfigsError> {
        let mut guard = match stream_t {
            alsa::Direction::Playback => self.playback.lock(),
            alsa::Direction::Capture => self.capture.lock(),
        }
        .unwrap();

        let handle = match &*guard {
            None => {
                let handle = alsa::pcm::PCM::new(&self.name, stream_t, true)?;
                *guard = Some(handle);
                guard.as_ref().unwrap()
            }
            Some(h) => h,
        };

        let hw_params = alsa::pcm::HwParams::any(&handle)?;

        // TODO: check endianess
        const FORMATS: [(SampleFormat, alsa::pcm::Format); 3] = [
            //SND_PCM_FORMAT_S8,
            //SND_PCM_FORMAT_U8,
            (SampleFormat::I16, alsa::pcm::Format::S16LE),
            //SND_PCM_FORMAT_S16_BE,
            (SampleFormat::U16, alsa::pcm::Format::U16LE),
            //SND_PCM_FORMAT_U16_BE,
            //SND_PCM_FORMAT_S24_LE,
            //SND_PCM_FORMAT_S24_BE,
            //SND_PCM_FORMAT_U24_LE,
            //SND_PCM_FORMAT_U24_BE,
            //SND_PCM_FORMAT_S32_LE,
            //SND_PCM_FORMAT_S32_BE,
            //SND_PCM_FORMAT_U32_LE,
            //SND_PCM_FORMAT_U32_BE,
            (SampleFormat::F32, alsa::pcm::Format::FloatLE),
            //SND_PCM_FORMAT_FLOAT_BE,
            //SND_PCM_FORMAT_FLOAT64_LE,
            //SND_PCM_FORMAT_FLOAT64_BE,
            //SND_PCM_FORMAT_IEC958_SUBFRAME_LE,
            //SND_PCM_FORMAT_IEC958_SUBFRAME_BE,
            //SND_PCM_FORMAT_MU_LAW,
            //SND_PCM_FORMAT_A_LAW,
            //SND_PCM_FORMAT_IMA_ADPCM,
            //SND_PCM_FORMAT_MPEG,
            //SND_PCM_FORMAT_GSM,
            //SND_PCM_FORMAT_SPECIAL,
            //SND_PCM_FORMAT_S24_3LE,
            //SND_PCM_FORMAT_S24_3BE,
            //SND_PCM_FORMAT_U24_3LE,
            //SND_PCM_FORMAT_U24_3BE,
            //SND_PCM_FORMAT_S20_3LE,
            //SND_PCM_FORMAT_S20_3BE,
            //SND_PCM_FORMAT_U20_3LE,
            //SND_PCM_FORMAT_U20_3BE,
            //SND_PCM_FORMAT_S18_3LE,
            //SND_PCM_FORMAT_S18_3BE,
            //SND_PCM_FORMAT_U18_3LE,
            //SND_PCM_FORMAT_U18_3BE,
        ];

        let mut supported_formats = Vec::new();
        for &(sample_format, alsa_format) in FORMATS.iter() {
            if hw_params.test_format(alsa_format).is_ok() {
                supported_formats.push(sample_format);
            }
        }

        let min_rate = hw_params.get_rate_min()?;
        let max_rate = hw_params.get_rate_max()?;

        let sample_rates = if min_rate == max_rate || hw_params.test_rate(min_rate + 1).is_ok() {
            vec![(min_rate, max_rate)]
        } else {
            const RATES: [libc::c_uint; 13] = [
                5512, 8000, 11025, 16000, 22050, 32000, 44100, 48000, 64000, 88200, 96000, 176400,
                192000,
            ];

            let mut rates = Vec::new();
            for &rate in RATES.iter() {
                if hw_params.test_rate(rate).is_ok() {
                    rates.push((rate, rate));
                }
            }

            if rates.len() == 0 {
                vec![(min_rate, max_rate)]
            } else {
                rates
            }
        };

        let min_channels = hw_params.get_channels_min()?;
        let max_channels = hw_params.get_channels_max()?;

        let max_channels = cmp::min(max_channels, 32); // TODO: limiting to 32 channels or too much stuff is returned
        let supported_channels = (min_channels..max_channels + 1)
            .filter_map(|num| {
                if hw_params.test_channels(num).is_ok() {
                    Some(num as ChannelCount)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let mut output = Vec::with_capacity(
            supported_formats.len() * supported_channels.len() * sample_rates.len(),
        );
        for &sample_format in supported_formats.iter() {
            for channels in supported_channels.iter() {
                for &(min_rate, max_rate) in sample_rates.iter() {
                    output.push(SupportedStreamConfigRange {
                        channels: channels.clone(),
                        min_sample_rate: SampleRate(min_rate as u32),
                        max_sample_rate: SampleRate(max_rate as u32),
                        sample_format: sample_format,
                    });
                }
            }
        }

        Ok(output.into_iter())
    }

    fn supported_input_configs(
        &self,
    ) -> Result<SupportedInputConfigs, SupportedStreamConfigsError> {
        self.supported_configs(alsa::Direction::Capture)
    }

    fn supported_output_configs(
        &self,
    ) -> Result<SupportedOutputConfigs, SupportedStreamConfigsError> {
        self.supported_configs(alsa::Direction::Playback)
    }

    // ALSA does not offer default stream formats, so instead we compare all supported formats by
    // the `SupportedStreamConfigRange::cmp_default_heuristics` order and select the greatest.
    fn default_config(
        &self,
        stream_t: alsa::Direction,
    ) -> Result<SupportedStreamConfig, DefaultStreamConfigError> {
        let mut formats: Vec<_> = {
            match self.supported_configs(stream_t) {
                Err(SupportedStreamConfigsError::DeviceNotAvailable) => {
                    return Err(DefaultStreamConfigError::DeviceNotAvailable);
                }
                Err(SupportedStreamConfigsError::InvalidArgument) => {
                    // this happens sometimes when querying for input and output capabilities but
                    // the device supports only one
                    return Err(DefaultStreamConfigError::StreamTypeNotSupported);
                }
                Err(SupportedStreamConfigsError::BackendSpecific { err }) => {
                    return Err(err.into());
                }
                Ok(fmts) => fmts.collect(),
            }
        };

        formats.sort_by(|a, b| a.cmp_default_heuristics(b));

        match formats.into_iter().last() {
            Some(f) => {
                let min_r = f.min_sample_rate;
                let max_r = f.max_sample_rate;
                let mut format = f.with_max_sample_rate();
                const HZ_44100: SampleRate = SampleRate(44_100);
                if min_r <= HZ_44100 && HZ_44100 <= max_r {
                    format.sample_rate = HZ_44100;
                }
                Ok(format)
            }
            None => Err(DefaultStreamConfigError::StreamTypeNotSupported),
        }
    }

    fn default_input_config(&self) -> Result<SupportedStreamConfig, DefaultStreamConfigError> {
        self.default_config(alsa::Direction::Capture)
    }

    fn default_output_config(&self) -> Result<SupportedStreamConfig, DefaultStreamConfigError> {
        self.default_config(alsa::Direction::Playback)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StreamState {
    Active,
    Paused,
    Dropping,
}

struct StreamInner {
    // The ALSA channel.
    channel: alsa::pcm::PCM,

    // Format of the samples.
    sample_format: SampleFormat,

    // Number of channels, ie. number of samples per frame.
    num_channels: u16,

    // Minimum number of samples to put in the buffer.
    period_len: usize,

    // Whether the stream is active, paused, or being dropped.
    state: (Mutex<StreamState>, Condvar),
}

// Assume that the ALSA library is built with thread safe option.
unsafe impl Sync for StreamInner {}

pub struct Stream {
    /// The high-priority audio processing thread calling callbacks.
    /// Option used for moving out in destructor.
    thread: Option<JoinHandle<()>>,

    /// Handle to the underlying stream for playback controls.
    inner: Arc<StreamInner>,
}

fn input_stream_worker(
    stream: &StreamInner,
    data_callback: &mut (dyn FnMut(&Data) + Send + 'static),
    error_callback: &mut (dyn FnMut(StreamError) + Send + 'static),
) {
    match stream.sample_format {
        SampleFormat::I16 => match stream.channel.io_i16() {
            Ok(io) => input_stream_worker_io(stream, io, data_callback, error_callback),
            Err(err) => error_callback(err.into()),
        },
        SampleFormat::U16 => match stream.channel.io_u16() {
            Ok(io) => input_stream_worker_io(stream, io, data_callback, error_callback),
            Err(err) => error_callback(err.into()),
        },
        SampleFormat::F32 => match stream.channel.io_f32() {
            Ok(io) => input_stream_worker_io(stream, io, data_callback, error_callback),
            Err(err) => error_callback(err.into()),
        },
    }
}

fn input_stream_worker_io<T: Default + Copy>(
    stream: &StreamInner,
    io: alsa::pcm::IO<'_, T>,
    data_callback: &mut (dyn FnMut(&Data) + Send + 'static),
    error_callback: &mut (dyn FnMut(StreamError) + Send + 'static),
) {
    let channels = stream.num_channels as usize;
    let mut buffer = vec![T::default(); stream.period_len];
    let data = unsafe {
        Data::from_parts(
            buffer.as_mut_ptr() as *mut (),
            buffer.len(),
            stream.sample_format,
        )
    };
    loop {
        let stream_state = {
            let mut guard = stream.state.0.lock().unwrap();

            // When pausing a Capture stream, drop any data still in the driver's buffer
            if *guard == StreamState::Paused && stream.channel.state() == alsa::pcm::State::Running
            {
                let _ = stream.channel.drop();
            }

            // If paused, block until the state changes
            while *guard == StreamState::Paused {
                guard = stream.state.1.wait(guard).unwrap();
            }
            *guard
        };

        match stream_state {
            StreamState::Active => {
                // Fill buffer from the stream
                let mut buf = &mut *buffer;
                while buf.len() > 0 {
                    match io
                        .readi(buf)
                        .or_else(|err| handle_stream_io_error(stream, err, error_callback))
                    {
                        Ok(frames) => buf = &mut buf[(frames * channels)..],
                        Err(()) => return,
                    }
                }

                // Give data to the callback
                data_callback(&data);
            }
            StreamState::Dropping => return,
            StreamState::Paused => unreachable!(),
        }
    }
}

fn output_stream_worker(
    stream: &StreamInner,
    data_callback: &mut (dyn FnMut(&mut Data) + Send + 'static),
    error_callback: &mut (dyn FnMut(StreamError) + Send + 'static),
) {
    match stream.sample_format {
        SampleFormat::I16 => match stream.channel.io_i16() {
            Ok(io) => output_stream_worker_io(stream, io, data_callback, error_callback),
            Err(err) => error_callback(err.into()),
        },
        SampleFormat::U16 => match stream.channel.io_u16() {
            Ok(io) => output_stream_worker_io(stream, io, data_callback, error_callback),
            Err(err) => error_callback(err.into()),
        },
        SampleFormat::F32 => match stream.channel.io_f32() {
            Ok(io) => output_stream_worker_io(stream, io, data_callback, error_callback),
            Err(err) => error_callback(err.into()),
        },
    }
}

fn output_stream_worker_io<T: Copy + Default>(
    stream: &StreamInner,
    io: alsa::pcm::IO<'_, T>,
    data_callback: &mut (dyn FnMut(&mut Data) + Send + 'static),
    error_callback: &mut (dyn FnMut(StreamError) + Send + 'static),
) {
    let channels = stream.num_channels as usize;
    let mut buffer = vec![T::default(); stream.period_len];
    let mut data = unsafe {
        Data::from_parts(
            buffer.as_mut_ptr() as *mut (),
            buffer.len(),
            stream.sample_format,
        )
    };

    loop {
        let stream_state = {
            let mut guard = stream.state.0.lock().unwrap();

            // When pausing a Playback stream, allow data in the driver's buffer to continue to play until consumed
            if *guard == StreamState::Paused && stream.channel.state() == alsa::pcm::State::Running
            {
                let _ = stream.channel.drain();
            }

            // If paused, block until the state changes
            while *guard == StreamState::Paused {
                guard = stream.state.1.wait(guard).unwrap();
            }
            *guard
        };

        match stream_state {
            StreamState::Active => {
                // Get data from the callback
                data_callback(&mut data);

                // Write the whole buffer to the stream
                let mut buf = &*buffer;
                while buf.len() > 0 {
                    match io
                        .writei(buf)
                        .or_else(|err| handle_stream_io_error(stream, err, error_callback))
                    {
                        Ok(frames) => buf = &buf[(frames * channels)..],
                        Err(()) => return,
                    }
                }
            }
            StreamState::Dropping => return,
            StreamState::Paused => unreachable!(),
        }
    }
}

fn handle_stream_io_error(
    stream: &StreamInner,
    err: alsa::Error,
    error_callback: &mut (dyn FnMut(StreamError) + Send + 'static),
) -> Result<usize, ()> {
    if let Some(errno) = err.errno() {
        match errno {
            nix::errno::Errno::EAGAIN => {
                stream.channel.wait(Some(100)).ok();
                Ok(0)
            }
            nix::errno::Errno::EBADFD => match stream.channel.prepare() {
                Ok(()) => Ok(0),
                Err(_) => {
                    error_callback(err.into());
                    Err(())
                }
            },
            nix::errno::Errno::EPIPE | nix::errno::Errno::EINTR | nix::errno::Errno::ESTRPIPE => {
                match stream.channel.try_recover(err, true) {
                    Ok(()) => {
                        error_callback(
                            BackendSpecificError {
                                description: format!("I/O error: {} (recovered)", errno),
                            }
                            .into(),
                        );
                        Ok(0)
                    }
                    Err(err) => {
                        error_callback(err.into());
                        Err(())
                    }
                }
            }
            _ => {
                let stream_state = { *stream.state.0.lock().unwrap() };
                if stream_state != StreamState::Dropping {
                    error_callback(err.into());
                }
                Err(())
            }
        }
    } else {
        error_callback(err.into());
        Err(())
    }
}

impl Stream {
    fn new_input<D, E>(
        inner: Arc<StreamInner>,
        mut data_callback: D,
        mut error_callback: E,
    ) -> Stream
    where
        D: FnMut(&Data) + Send + 'static,
        E: FnMut(StreamError) + Send + 'static,
    {
        // Clone the handle for passing into worker thread.
        let stream = inner.clone();
        let thread = thread::spawn(move || {
            input_stream_worker(&*stream, &mut data_callback, &mut error_callback);
        });
        Stream {
            thread: Some(thread),
            inner,
        }
    }

    fn new_output<D, E>(
        inner: Arc<StreamInner>,
        mut data_callback: D,
        mut error_callback: E,
    ) -> Stream
    where
        D: FnMut(&mut Data) + Send + 'static,
        E: FnMut(StreamError) + Send + 'static,
    {
        // Clone the handle for passing into worker thread.
        let stream = inner.clone();
        let thread = thread::spawn(move || {
            output_stream_worker(&*stream, &mut data_callback, &mut error_callback);
        });
        Stream {
            thread: Some(thread),
            inner,
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        {
            let mut guard = self.inner.state.0.lock().unwrap();
            *guard = StreamState::Dropping;
            self.inner.state.1.notify_one();
        }
        let _ = self.inner.channel.drop();
        self.thread.take().unwrap().join().unwrap();
    }
}

impl StreamTrait for Stream {
    fn play(&self) -> Result<(), PlayStreamError> {
        let mut guard = self.inner.state.0.lock().unwrap();
        *guard = StreamState::Active;
        self.inner.state.1.notify_one();
        Ok(())
    }
    fn pause(&self) -> Result<(), PauseStreamError> {
        let mut guard = self.inner.state.0.lock().unwrap();
        *guard = StreamState::Paused;
        Ok(())
    }
}

fn set_hw_params_from_format<'a>(
    pcm_handle: &'a alsa::pcm::PCM,
    config: &StreamConfig,
    sample_format: SampleFormat,
) -> Result<alsa::pcm::HwParams<'a>, BackendSpecificError> {
    let hw_params = alsa::pcm::HwParams::any(pcm_handle)?;
    hw_params.set_access(alsa::pcm::Access::RWInterleaved)?;

    let sample_format = if cfg!(target_endian = "big") {
        match sample_format {
            SampleFormat::I16 => alsa::pcm::Format::S16BE,
            SampleFormat::U16 => alsa::pcm::Format::U16BE,
            SampleFormat::F32 => alsa::pcm::Format::FloatBE,
        }
    } else {
        match sample_format {
            SampleFormat::I16 => alsa::pcm::Format::S16LE,
            SampleFormat::U16 => alsa::pcm::Format::U16LE,
            SampleFormat::F32 => alsa::pcm::Format::FloatLE,
        }
    };

    hw_params.set_format(sample_format)?;
    hw_params.set_rate(config.sample_rate.0, alsa::ValueOr::Nearest)?;
    hw_params.set_channels(config.channels as u32)?;

    // If this isn't set manually a overlarge buffer may be used causing audio delay
    hw_params.set_buffer_time_near(100_000, alsa::ValueOr::Nearest)?;

    pcm_handle.hw_params(&hw_params)?;

    Ok(hw_params)
}

fn set_sw_params_from_format(
    pcm_handle: &alsa::pcm::PCM,
    config: &StreamConfig,
    stream_type: alsa::Direction,
) -> Result<usize, BackendSpecificError> {
    let sw_params = pcm_handle.sw_params_current()?;

    let period_len = {
        let (buffer, period) = pcm_handle.get_params()?;
        if buffer == 0 {
            return Err(BackendSpecificError {
                description: "initialization resulted in a null buffer".to_string(),
            });
        }
        sw_params.set_avail_min(period as alsa::pcm::Frames)?;

        let start_threshold = match stream_type {
            alsa::Direction::Playback => buffer - period,
            alsa::Direction::Capture => 1,
        };
        sw_params.set_start_threshold(start_threshold as alsa::pcm::Frames)?;

        period as usize * config.channels as usize
    };

    pcm_handle.sw_params(&sw_params)?;

    Ok(period_len)
}

impl From<alsa::Error> for BackendSpecificError {
    fn from(err: alsa::Error) -> Self {
        BackendSpecificError {
            description: err.to_string(),
        }
    }
}

impl From<alsa::pcm::Status> for BackendSpecificError {
    fn from(status: alsa::pcm::Status) -> Self {
        BackendSpecificError {
            description: match alsa::Output::buffer_open() {
                Ok(mut output) => {
                    status.dump(&mut output).expect("ALSA status dump failed");
                    output.buffer_string(|bytes| String::from_utf8_lossy(bytes).into_owned())
                }
                Err(err) => err.to_string(),
            },
        }
    }
}

impl From<alsa::Error> for BuildStreamError {
    fn from(err: alsa::Error) -> Self {
        match err.errno() {
            Some(nix::errno::Errno::EBUSY) => BuildStreamError::DeviceNotAvailable,
            Some(nix::errno::Errno::EINVAL) => BuildStreamError::InvalidArgument,
            _ => {
                let err: BackendSpecificError = err.into();
                err.into()
            }
        }
    }
}

impl From<alsa::Error> for SupportedStreamConfigsError {
    fn from(err: alsa::Error) -> Self {
        match err.errno() {
            Some(nix::errno::Errno::ENOENT) | Some(nix::errno::Errno::EBUSY) => {
                SupportedStreamConfigsError::DeviceNotAvailable
            }
            Some(nix::errno::Errno::EINVAL) => SupportedStreamConfigsError::InvalidArgument,
            _ => {
                let err: BackendSpecificError = err.into();
                err.into()
            }
        }
    }
}

impl From<alsa::Error> for PlayStreamError {
    fn from(err: alsa::Error) -> Self {
        let err: BackendSpecificError = err.into();
        err.into()
    }
}

impl From<alsa::Error> for PauseStreamError {
    fn from(err: alsa::Error) -> Self {
        let err: BackendSpecificError = err.into();
        err.into()
    }
}

impl From<alsa::Error> for StreamError {
    fn from(err: alsa::Error) -> Self {
        let err: BackendSpecificError = err.into();
        err.into()
    }
}
