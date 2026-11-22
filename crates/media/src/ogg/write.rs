//! Ogg writer, written from RFC 3533 (pages and lacing) and RFC 7845 (the Ogg Opus mapping).
//!
//! [`Writer`] lays packets of one logical bitstream out in pages: each page carries up to 255
//! lacing values, a packet longer than a page goes on in the next ones (the "continued" flag), and
//! a page's granule position is the one of the last packet that ends on it (-1 when none does).
//! The first page is marked as the start of the stream, the last one as its end, and every page
//! carries its sequence number and CRC-32. [`opus_tags`] builds the comment header an Ogg Opus
//! stream needs after its `OpusHead`.

use std::io::{self, Write};

/// Header type flag: the page's first packet began on an earlier page.
const CONTINUED: u8 = 0x01;
/// Header type flag: first page of the logical bitstream.
const BEGIN: u8 = 0x02;
/// Header type flag: last page of the logical bitstream.
const END: u8 = 0x04;
/// Pages are closed once their body reaches this many bytes (a packet's end may go a little past).
const PAGE_TARGET: usize = 4096;

/// Writes one logical Ogg bitstream to `W`, page by page.
///
/// ```
/// use craft_media::ogg::{Writer, open};
///
/// let mut w = Writer::new(Vec::new(), 0x5eed);
/// w.packet(b"header", 0)?;
/// w.flush_page()?; // headers sit on pages of their own
/// w.packet(&[7u8; 600], 960)?;
/// let bytes = w.finish()?;
/// let file = open(&bytes[..])?;
/// assert_eq!(file.streams.len(), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct Writer<W: Write> {
    out: W,
    serial: u32,
    sequence: u32,
    /// The next page is the stream's first.
    first: bool,
    /// Lacing values and body of the page being filled.
    lacing: Vec<u8>,
    body: Vec<u8>,
    /// The page being filled starts with the rest of a packet begun on the previous page.
    continued: bool,
    /// Granule position of the last packet that ended on the page being filled (-1: none did).
    granule: i64,
    /// Granule position of the last packet written, for the end page.
    last_granule: i64,
}

impl<W: Write> Writer<W> {
    /// A writer for the logical bitstream with serial number `serial`.
    pub fn new(out: W, serial: u32) -> Writer<W> {
        Writer { out, serial, sequence: 0, first: true, lacing: Vec::new(), body: Vec::new(), continued: false, granule: -1, last_granule: 0 }
    }

    /// Add one packet whose last sample is at `granule` (for Opus: 48 kHz samples since the start,
    /// pre-skip included; 0 for header packets). Full pages are written as the packet fills them.
    pub fn packet(&mut self, data: &[u8], granule: i64) -> io::Result<()> {
        let mut rest = data;
        loop {
            if self.lacing.len() == 255 {
                // The page is full in the middle of the packet: it goes on in the next one.
                self.write_page(0)?;
                self.continued = true;
            }
            let take = rest.len().min(255);
            self.lacing.push(take as u8);
            self.body.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            // A lacing value below 255 ends the packet (a packet of a multiple of 255 bytes ends with a 0).
            if take < 255 {
                break;
            }
        }
        self.granule = granule;
        self.last_granule = granule;
        if self.body.len() >= PAGE_TARGET {
            self.write_page(0)?;
        }
        Ok(())
    }

    /// Write the page being filled now, if it holds anything, so that the next packet starts a new
    /// page (RFC 7845 asks for this after each header packet).
    pub fn flush_page(&mut self) -> io::Result<()> {
        if !self.lacing.is_empty() {
            self.write_page(0)?;
        }
        Ok(())
    }

    /// Write the last page, marked as the end of the stream, and give back the output.
    pub fn finish(mut self) -> io::Result<W> {
        if self.lacing.is_empty() {
            // Everything is on pages already: an empty page carries the end mark.
            self.granule = self.last_granule;
        }
        self.write_page(END)?;
        self.out.flush()?;
        Ok(self.out)
    }

    fn write_page(&mut self, flags: u8) -> io::Result<()> {
        let mut flags = flags;
        if std::mem::take(&mut self.first) {
            flags |= BEGIN;
        }
        if std::mem::take(&mut self.continued) {
            flags |= CONTINUED;
        }
        let mut page = Vec::with_capacity(27 + self.lacing.len() + self.body.len());
        page.extend_from_slice(b"OggS");
        page.push(0); // stream structure version
        page.push(flags);
        page.extend_from_slice(&self.granule.to_le_bytes());
        page.extend_from_slice(&self.serial.to_le_bytes());
        page.extend_from_slice(&self.sequence.to_le_bytes());
        page.extend_from_slice(&[0; 4]); // CRC, filled in below
        page.push(self.lacing.len() as u8);
        page.extend_from_slice(&self.lacing);
        page.extend_from_slice(&self.body);
        let crc = super::crc32(&page);
        page[22..26].copy_from_slice(&crc.to_le_bytes());
        self.out.write_all(&page)?;
        self.sequence = self.sequence.wrapping_add(1);
        self.lacing.clear();
        self.body.clear();
        self.granule = -1;
        Ok(())
    }
}

/// The `OpusTags` comment header (RFC 7845 §5.2) with vendor string `vendor` and no user comments.
pub fn opus_tags(vendor: &str) -> Vec<u8> {
    let mut h = Vec::with_capacity(16 + vendor.len());
    h.extend_from_slice(b"OpusTags");
    h.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    h.extend_from_slice(vendor.as_bytes());
    h.extend_from_slice(&0u32.to_le_bytes());
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ogg::{Codec, open};

    fn opus_head() -> Vec<u8> {
        let mut h = b"OpusHead".to_vec();
        h.extend_from_slice(&[1, 2]);
        h.extend_from_slice(&312u16.to_le_bytes());
        h.extend_from_slice(&48_000u32.to_le_bytes());
        h.extend_from_slice(&[0, 0, 0]);
        h
    }

    /// What is written reads back with the crate's own demuxer: one Opus stream, its headers apart
    /// from its audio, every packet byte for byte (also one longer than a page), each page's CRC
    /// accepted, and the granule position of the last packet on the last page.
    #[test]
    fn written_pages_read_back_as_the_same_packets() {
        let mut w = Writer::new(Vec::new(), 42);
        w.packet(&opus_head(), 0).unwrap();
        w.flush_page().unwrap();
        w.packet(&opus_tags("craft"), 0).unwrap();
        w.flush_page().unwrap();
        // A 20 ms CELT packet (TOC 0xFC: config 31, stereo), many of them, one 70 000 bytes long.
        let packets: Vec<Vec<u8>> = (0..200usize)
            .map(|i| {
                let len = if i == 77 { 70_000 } else { 100 + i % 50 };
                let mut p = vec![0xFC];
                p.extend((0..len).map(|j| (i * 7 + j) as u8));
                p
            })
            .collect();
        for (i, p) in packets.iter().enumerate() {
            w.packet(p, 312 + 960 * (i as i64 + 1)).unwrap();
        }
        let bytes = w.finish().unwrap();
        let file = open(&bytes[..]).unwrap();
        assert_eq!(file.streams.len(), 1);
        assert!(file.warnings.is_empty(), "every page reads clean: {:?}", file.warnings);
        let s = file.stream_of(Codec::Opus).expect("an Opus stream");
        assert_eq!(file.streams[s].headers.len(), 2, "OpusHead and OpusTags");
        assert_eq!(file.streams[s].last_granule, Some(312 + 960 * packets.len() as i64));
        let n = file.streams[s].packets.len();
        assert_eq!(n, packets.len(), "every audio packet, headers apart");
        for (i, p) in packets.iter().enumerate() {
            assert_eq!(&file.read_packet(&bytes[..], s, i).unwrap(), p, "packet {i}");
        }
        assert!(bytes.len() > 70_000 + 27 * (70_000 / (255 * 255) + 1), "the long packet spans pages");
    }

    /// A packet of exactly 255 bytes ends with a zero lacing value, and a stream whose packets are
    /// all on pages already still ends with an end-of-stream page.
    #[test]
    fn lacing_and_the_end_page() {
        let mut w = Writer::new(Vec::new(), 1);
        w.packet(&[9u8; 255], 5).unwrap();
        w.flush_page().unwrap();
        let bytes = w.finish().unwrap();
        // First page: one packet as the values 255, 0.
        assert_eq!(&bytes[26..29], &[2, 255, 0]);
        assert_eq!(bytes[5] & BEGIN, BEGIN);
        let last = bytes.windows(4).rposition(|x| x == b"OggS").unwrap();
        assert_eq!(bytes[last + 5] & END, END, "the last page ends the stream");
        assert_eq!(i64::from_le_bytes(bytes[last + 6..last + 14].try_into().unwrap()), 5);
    }
}
