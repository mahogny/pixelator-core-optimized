mod dropped_read_buffer;
mod extract;
mod processor;
mod qc;
mod read_layout;
mod row_buffer;
mod stats;

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::mpsc::{Receiver, sync_channel};
use std::thread::{self, JoinHandle};

use crate::common::io::write_record_batches_to_path;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use paraseq::fastq;
use paraseq::prelude::*;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use crate::common::{Assay, Panel};

use dropped_read_buffer::DroppedReadBuffer;
pub use extract::{DEFAULT_LBS_OVERLAP_MAX_EDITS, FilterConfig, LbsSearcher};
pub use processor::{PairedProcessor, SingleProcessor};
pub use read_layout::Strand;
pub use row_buffer::RowBuffer;
pub use stats::{EdgelistStats, Q30Statistics};

/// Accumulation threshold for worker threads before sending a RecordBatch.
///
/// Matches Parquet's default max row group size.
const BATCH_THRESHOLD: usize = 1_000_000;

/// Multiplier used to determine the bounded MPSC channel capacity (`CHANNEL_FACTOR * n_threads`).
///
/// Sizing logic:
/// - Shock absorption: if workers parse FASTQ much faster than the background thread can
///   compress Parquet. This bounded queue safely absorbs that speed mismatch.
/// - Backpressure: A factor of 4 keeps the queue large enough so workers rarely block,
///   but small enough to force workers to pause if the writer falls severely behind.
/// - RAM Safety: With 16 threads, a max queue of 64 batches (~28MB per 1M-row batch)
///   caps the queue's memory footprint around ~1.8GB, preventing OOM crashes on a 64GB system.
const CHANNEL_SIZE_FACTOR: usize = 4;

/// Rows per dropped-reads batch, far fewer than [`BATCH_THRESHOLD`] because each
/// row carries whole reads: a 2x150 bp pair with qualities is ~600 bytes against
/// the edgelist's 24. At this size a batch is ~30MB.
const DROPPED_BATCH_THRESHOLD: usize = 50_000;

/// Bounded channel capacity for dropped-read batches, per worker thread.
///
/// Smaller than [`CHANNEL_SIZE_FACTOR`] for the same reason the batches are: with
/// 16 threads, 16 queued batches plus one filling per worker caps the feature's
/// memory around ~1GB. A writer that falls behind applies backpressure instead.
const DROPPED_CHANNEL_SIZE_FACTOR: usize = 1;

type BoxedReader = Box<dyn Read + Send>;

/// FASTQ source for [`edgelist`]: single-end (forward or reverse) or paired-end.
pub enum FastqInput {
    SingleForward(BoxedReader),
    SingleReverse(BoxedReader),
    Paired { r1: BoxedReader, r2: BoxedReader },
}

/// Builds the edgelist Parquet from `input`.
///
/// Output is written to `{output_dir}/{sample_name}.parquet`. When
/// `save_dropped_reads` is set, every read the extractor discards is also written,
/// with the reason it was dropped, to `{output_dir}/{sample_name}.dropped_reads.parquet`.
///
/// `reads_processed` is incremented (off the per-read hot path) as reads are
/// consumed; callers can observe it from another thread to report progress.
///
/// Returns the run's tallies, from which the caller builds the sample report.
#[allow(clippy::too_many_arguments)]
pub fn edgelist<P: AsRef<Path>>(
    input: FastqInput,
    sample_name: String,
    assay: Assay,
    panel: Panel,
    output_dir: P,
    n_threads: usize,
    filters: FilterConfig,
    save_dropped_reads: bool,
    reads_processed: Arc<AtomicU64>,
) -> Result<EdgelistStats, Box<dyn std::error::Error>> {
    let output_dir = output_dir.as_ref();
    let path = output_dir.join(format!("{sample_name}.parquet"));
    edgelist_impl(
        input, sample_name, assay, panel, output_dir, n_threads, filters, save_dropped_reads, reads_processed,
        Box::new(move |rx: Receiver<RecordBatch>| {
            write_record_batches_to_path(path, RowBuffer::schema(), rx.into_iter(), None).map_err(|e| e.to_string())
        }),
    )
}

/// Consumer of the kept-row batches (runs on its own thread). [pixelator-rs patch]
pub type BatchSink = Box<dyn FnOnce(Receiver<RecordBatch>) -> Result<(), String> + Send>;

/// Like [`edgelist`], but the kept rows go to `sink` instead of a Parquet file (dropped reads,
/// if requested, are still written to `{output_dir}/{sample_name}.dropped_reads.parquet`).
/// [pixelator-rs patch]
#[allow(clippy::too_many_arguments)]
pub fn edgelist_to_sink<P: AsRef<Path>>(
    input: FastqInput,
    sample_name: String,
    assay: Assay,
    panel: Panel,
    output_dir: P,
    n_threads: usize,
    filters: FilterConfig,
    save_dropped_reads: bool,
    reads_processed: Arc<AtomicU64>,
    sink: BatchSink,
) -> Result<EdgelistStats, Box<dyn std::error::Error>> {
    edgelist_impl(input, sample_name, assay, panel, output_dir.as_ref(), n_threads, filters, save_dropped_reads, reads_processed, sink)
}

#[allow(clippy::too_many_arguments)]
fn edgelist_impl(
    input: FastqInput,
    sample_name: String,
    assay: Assay,
    panel: Panel,
    output_dir: &Path,
    n_threads: usize,
    filters: FilterConfig,
    save_dropped_reads: bool,
    reads_processed: Arc<AtomicU64>,
    sink: BatchSink,
) -> Result<EdgelistStats, Box<dyn std::error::Error>> {
    let (tx, rx) = sync_channel(CHANNEL_SIZE_FACTOR * n_threads);
    let handle = thread::spawn(move || sink(rx));

    let (dropped_tx, dropped_handle) = if save_dropped_reads {
        let (tx, rx) = sync_channel(DROPPED_CHANNEL_SIZE_FACTOR * n_threads);
        // Unlike the edgelist's packed integers, raw sequence and quality text
        // compresses several-fold, which is worth the writer thread's time.
        let properties = WriterProperties::builder()
            .set_compression(Compression::ZSTD(ZstdLevel::default()))
            .build();
        let handle = spawn_parquet_writer(
            output_dir.join(format!("{sample_name}.dropped_reads.parquet")),
            DroppedReadBuffer::schema(),
            rx,
            Some(properties),
        );
        (Some(tx), Some(handle))
    } else {
        (None, None)
    };
    let (pipeline, stats) = match input {
        FastqInput::SingleForward(r) => {
            let mut proc =
                SingleProcessor::new(tx, BATCH_THRESHOLD, assay, panel, Strand::Forward, filters)
                    .with_progress_counter(reads_processed)
                    .with_dropped_reads(dropped_tx, DROPPED_BATCH_THRESHOLD);
            (
                fastq::Reader::new(r).process_parallel(&mut proc, n_threads),
                proc.into_stats(),
            )
        }
        FastqInput::SingleReverse(r) => {
            let mut proc =
                SingleProcessor::new(tx, BATCH_THRESHOLD, assay, panel, Strand::Reverse, filters)
                    .with_progress_counter(reads_processed)
                    .with_dropped_reads(dropped_tx, DROPPED_BATCH_THRESHOLD);
            (
                fastq::Reader::new(r).process_parallel(&mut proc, n_threads),
                proc.into_stats(),
            )
        }
        FastqInput::Paired { r1, r2 } => {
            let mut proc = PairedProcessor::new(tx, BATCH_THRESHOLD, assay, panel, filters)
                .with_progress_counter(reads_processed)
                .with_dropped_reads(dropped_tx, DROPPED_BATCH_THRESHOLD);
            (
                fastq::Reader::new(r1).process_parallel_paired(
                    fastq::Reader::new(r2),
                    &mut proc,
                    n_threads,
                ),
                proc.into_stats(),
            )
        }
    };

    // `into_stats` consumed the processor and with it every sender, which is what
    // ends each writer's receive loop.
    let writer = handle.join().expect("parquet writer thread panicked");
    let dropped_writer = dropped_handle
        .map(|h| h.join().expect("dropped-reads writer thread panicked"))
        .transpose();

    // Surface all errors that may have occured during processing and writing.
    pipeline?;
    writer?;
    dropped_writer?;

    Ok(stats)
}

/// Spawns a thread that writes every batch received on `rx` to a Parquet file at
/// `path`, finishing once all senders are gone.
fn spawn_parquet_writer(
    path: PathBuf,
    schema: SchemaRef,
    rx: Receiver<RecordBatch>,
    properties: Option<WriterProperties>,
) -> JoinHandle<Result<(), String>> {
    thread::spawn(move || {
        write_record_batches_to_path(path, schema, rx.into_iter(), properties)
            .map_err(|e| e.to_string())
    })
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Cursor;

    use arrow::array::{Array, ArrayAccessor, DictionaryArray, StringArray};
    use arrow::datatypes::Int32Type;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

    use super::*;
    use crate::common::PanelMarker;
    use crate::test_fixtures::{
        OwnedRecord, ReadMixConfig, generate_single_reads, proxiome_v2_assay,
        proxiome_v2_marker_sequences, proxiome_v2_panel, proxiome_v2_read_pairs,
    };

    /// Serializes one side of a set of read pairs into an in-memory FASTQ buffer.
    ///
    /// The generated fixtures carry an empty `id`, so a synthetic `read{i}` name
    /// is emitted to keep each record a well-formed 4-line FASTQ entry.
    fn records_to_fastq<'a>(records: impl Iterator<Item = &'a OwnedRecord>) -> Vec<u8> {
        let mut out = Vec::new();
        for (i, rec) in records.enumerate() {
            out.extend_from_slice(format!("@read{i}\n").as_bytes());
            out.extend_from_slice(&rec.seq);
            out.push(b'\n');
            out.extend_from_slice(b"+\n");
            out.extend_from_slice(&rec.qual);
            out.push(b'\n');
        }
        out
    }

    const N: usize = 1000;

    const EDGELIST_COLUMNS: [&str; 6] = [
        "marker_1",
        "marker_2",
        "umi1",
        "umi2",
        "marker_sequence_1",
        "marker_sequence_2",
    ];

    /// A fully valid mix (no N-drops, no unknown pids) with reads long enough to
    /// span the whole amplicon, so every generated read produces exactly one row.
    fn valid_mix() -> ReadMixConfig {
        ReadMixConfig {
            frac_has_n: 0.0,
            frac_unknown_pid: 0.0,
            read_len: 150,
        }
    }

    /// What [`run_edgelist_to_parquet`] read back from the edgelist Parquet.
    struct ParquetContents {
        stats: EdgelistStats,
        field_names: Vec<String>,
        total_rows: usize,
        /// Rows per marker id in each of the `marker_1` / `marker_2` columns.
        written: [HashMap<String, usize>; 2],
        /// Distinct sequences stored alongside each marker id, in each of the
        /// `marker_sequence_1` / `marker_sequence_2` columns.
        sequences: [HashMap<String, HashSet<String>>; 2],
    }

    /// Runs `edgelist` on `input` and reads the resulting Parquet back.
    fn run_edgelist_to_parquet(input: FastqInput) -> ParquetContents {
        let tmpdir = tempfile::tempdir().expect("create tempdir");
        let sample_name = "sample".to_string();

        let stats = edgelist(
            input,
            sample_name.clone(),
            proxiome_v2_assay(),
            proxiome_v2_panel(),
            tmpdir.path(),
            2,
            FilterConfig::default(),
            false,
            Arc::new(AtomicU64::new(0)),
        )
        .expect("edgelist run");

        let parquet_path = tmpdir.path().join(format!("{sample_name}.parquet"));
        let file = File::open(&parquet_path).expect("open parquet");
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet reader");

        let field_names: Vec<String> = builder
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();

        let mut written: [HashMap<String, usize>; 2] = Default::default();
        let mut sequences: [HashMap<String, HashSet<String>>; 2] = Default::default();
        let mut total_rows = 0;
        for batch in builder.build().expect("build parquet reader") {
            let batch = batch.expect("read record batch");
            total_rows += batch.num_rows();
            let sides = [
                ("marker_1", "marker_sequence_1"),
                ("marker_2", "marker_sequence_2"),
            ];
            for (side, (column, sequence_column)) in sides.into_iter().enumerate() {
                let dict = batch
                    .column_by_name(column)
                    .expect("marker column")
                    .as_any()
                    .downcast_ref::<DictionaryArray<Int32Type>>()
                    .expect("dictionary-encoded marker column");
                let ids = dict
                    .values()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("string dictionary values");
                let sequence_dict = batch
                    .column_by_name(sequence_column)
                    .expect("sequence column")
                    .as_any()
                    .downcast_ref::<DictionaryArray<Int32Type>>()
                    .expect("dictionary-encoded sequence column")
                    .downcast_dict::<StringArray>()
                    .expect("string dictionary values");
                for (row, key) in dict.keys().iter().enumerate() {
                    let id = ids.value(key.expect("non-null key") as usize);
                    *written[side].entry(id.to_string()).or_insert(0) += 1;
                    sequences[side]
                        .entry(id.to_string())
                        .or_default()
                        .insert(sequence_dict.value(row).to_string());
                }
            }
        }

        ParquetContents {
            stats,
            field_names,
            total_rows,
            written,
            sequences,
        }
    }

    /// Asserts every marker id was stored with exactly its panel barcode:
    /// `sequence_1` in `marker_sequence_1`, `sequence_2` in `marker_sequence_2`.
    fn assert_sequences_match_panel(sequences: &[HashMap<String, HashSet<String>>; 2]) {
        let panel = proxiome_v2_panel();
        let markers: HashMap<&str, &PanelMarker> = panel
            .markers()
            .iter()
            .map(|m| (m.marker_id.as_str(), m))
            .collect();
        for (side, stored) in sequences.iter().enumerate() {
            for (id, stored_sequences) in stored {
                let marker = markers[id.as_str()];
                let expected = if side == 0 {
                    &marker.sequence_1
                } else {
                    &marker.sequence_2
                };
                assert_eq!(
                    stored_sequences,
                    &HashSet::from_iter([expected.clone()]),
                    "marker_sequence_{} for {id}",
                    side + 1
                );
            }
        }
    }

    fn assert_marker_tallies_match_parquet(
        stats: &EdgelistStats,
        written: &[HashMap<String, usize>; 2],
    ) {
        let panel = proxiome_v2_panel();
        let reported = [
            stats.marker1_by_frequency(&panel),
            stats.marker2_by_frequency(&panel),
        ];
        for (side, ranked) in reported.iter().enumerate() {
            assert_eq!(ranked.len(), panel.markers().len());
            assert!(
                ranked.windows(2).all(|w| w[0].1 >= w[1].1),
                "marker_{} tally is not ranked by count",
                side + 1
            );
            for &(id, count) in ranked {
                let in_parquet = written[side].get(id).copied().unwrap_or(0);
                assert_eq!(count, in_parquet, "marker_{} count for {id}", side + 1);
            }
        }
    }

    /// Runs `edgelist` on `input` and asserts it writes a schema-correct Parquet
    /// with exactly `expected_rows` rows (one per valid read), and that the
    /// returned stats describe that same run.
    fn assert_writes_one_row_per_read(input: FastqInput, expected_rows: usize) {
        let ParquetContents {
            stats,
            field_names,
            total_rows,
            written,
            sequences,
        } = run_edgelist_to_parquet(input);
        assert_eq!(field_names, EDGELIST_COLUMNS);
        assert_marker_tallies_match_parquet(&stats, &written);
        assert_sequences_match_panel(&sequences);

        assert_eq!(
            total_rows, expected_rows,
            "every valid read should produce one row"
        );

        assert_eq!(stats.total_reads(), expected_rows);
        assert_eq!(stats.total_kept_reads(), total_rows);
        assert!(
            stats
                .dropped_by_frequency()
                .iter()
                .all(|&(_, count)| count == 0),
            "a fully valid mix should drop nothing"
        );

        // The fixtures fill quality with a uniform `b'I'` (Phred 40), so every
        // base clears Q30. Anything below 1.0 means a region was measured over
        // the wrong bases; a flat 0.0 means it was never measured at all.
        let q30 = stats.q30();
        assert_eq!(q30.total, 1.0);
        assert_eq!(q30.umi1, 1.0);
        assert_eq!(q30.pid1, 1.0);
        assert_eq!(q30.pid2, 1.0);
        assert_eq!(q30.umi2, 1.0);
    }

    #[test]
    fn edgelist_end_to_end_paired() {
        let pairs = proxiome_v2_read_pairs(N, valid_mix(), 0);
        let input = FastqInput::Paired {
            r1: Box::new(Cursor::new(records_to_fastq(
                pairs.iter().map(|(r1, _)| r1),
            ))),
            r2: Box::new(Cursor::new(records_to_fastq(
                pairs.iter().map(|(_, r2)| r2),
            ))),
        };
        assert_writes_one_row_per_read(input, N);
    }

    #[test]
    fn edgelist_end_to_end_single_forward() {
        let reads = generate_single_reads(
            &proxiome_v2_assay(),
            &proxiome_v2_marker_sequences(),
            N,
            valid_mix(),
            0,
            Strand::Forward,
        );
        let input =
            FastqInput::SingleForward(Box::new(Cursor::new(records_to_fastq(reads.iter()))));
        assert_writes_one_row_per_read(input, N);
    }

    #[test]
    fn edgelist_end_to_end_single_reverse() {
        let reads = generate_single_reads(
            &proxiome_v2_assay(),
            &proxiome_v2_marker_sequences(),
            N,
            valid_mix(),
            0,
            Strand::Reverse,
        );
        let input =
            FastqInput::SingleReverse(Box::new(Cursor::new(records_to_fastq(reads.iter()))));
        assert_writes_one_row_per_read(input, N);
    }

    /// A tenth of the reads carry an injected `N`; nothing else in this mix can
    /// drop a read, so the whole shortfall must be reported under `has_n`.
    #[test]
    fn edgelist_reports_dropped_reads_under_their_reason() {
        let mix = ReadMixConfig {
            frac_has_n: 0.1,
            frac_unknown_pid: 0.0,
            read_len: 150,
        };
        let pairs = proxiome_v2_read_pairs(N, mix, 0);
        let input = FastqInput::Paired {
            r1: Box::new(Cursor::new(records_to_fastq(
                pairs.iter().map(|(r1, _)| r1),
            ))),
            r2: Box::new(Cursor::new(records_to_fastq(
                pairs.iter().map(|(_, r2)| r2),
            ))),
        };

        let ParquetContents {
            stats,
            field_names,
            total_rows,
            written,
            sequences,
        } = run_edgelist_to_parquet(input);
        assert_eq!(field_names, EDGELIST_COLUMNS);
        assert_marker_tallies_match_parquet(&stats, &written);
        assert_sequences_match_panel(&sequences);

        assert_eq!(stats.total_reads(), N, "every read is counted");
        assert_eq!(stats.total_kept_reads(), total_rows);

        let dropped = stats.total_dropped_reads();
        assert!(dropped > 0, "this mix is meant to drop reads");

        let ranked = stats.dropped_by_frequency();
        assert_eq!(ranked[0], ("has_n", dropped));
        assert!(
            ranked[1..].iter().all(|&(_, count)| count == 0),
            "no other reason should fire: {ranked:?}"
        );

        // Dropped reads are still measured, so QC covers all N reads.
        assert_eq!(stats.q30().total, 1.0);
    }

    /// With `save_dropped_reads`, every dropped pair lands in the side Parquet,
    /// verbatim and under the reason the report tallies it by.
    #[test]
    fn edgelist_saves_dropped_reads_with_their_reason() {
        let mix = ReadMixConfig {
            frac_has_n: 0.1,
            frac_unknown_pid: 0.1,
            read_len: 150,
        };
        let pairs = proxiome_v2_read_pairs(N, mix, 0);
        let input = FastqInput::Paired {
            r1: Box::new(Cursor::new(records_to_fastq(
                pairs.iter().map(|(r1, _)| r1),
            ))),
            r2: Box::new(Cursor::new(records_to_fastq(
                pairs.iter().map(|(_, r2)| r2),
            ))),
        };

        let tmpdir = tempfile::tempdir().expect("create tempdir");
        let stats = edgelist(
            input,
            "sample".to_string(),
            proxiome_v2_assay(),
            proxiome_v2_panel(),
            tmpdir.path(),
            2,
            FilterConfig::default(),
            true,
            Arc::new(AtomicU64::new(0)),
        )
        .expect("edgelist run");

        let file = File::open(tmpdir.path().join("sample.dropped_reads.parquet"))
            .expect("open dropped-reads parquet");
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet reader");
        assert_eq!(
            builder.schema().as_ref(),
            DroppedReadBuffer::schema().as_ref()
        );

        let string_column = |batch: &RecordBatch, name: &str| -> StringArray {
            batch
                .column_by_name(name)
                .expect("column present")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("utf8 column")
                .clone()
        };

        let mut rows = 0;
        let mut per_reason: HashMap<String, usize> = HashMap::default();
        for batch in builder.build().expect("build parquet reader") {
            let batch = batch.expect("read record batch");
            let names = string_column(&batch, "read_name");
            let seq1 = string_column(&batch, "sequence");
            let qual1 = string_column(&batch, "quality");
            let seq2 = string_column(&batch, "sequence_2");
            let qual2 = string_column(&batch, "quality_2");
            let reasons = batch
                .column_by_name("reason")
                .expect("reason column")
                .as_any()
                .downcast_ref::<DictionaryArray<Int32Type>>()
                .expect("dictionary-encoded reason")
                .downcast_dict::<StringArray>()
                .expect("string dictionary values");

            for i in 0..batch.num_rows() {
                // `records_to_fastq` names each record after its input position.
                let index: usize = names.value(i)["read".len()..]
                    .parse()
                    .expect("read{i} name");
                let (r1, r2) = &pairs[index];
                assert_eq!(seq1.value(i).as_bytes(), r1.seq);
                assert_eq!(qual1.value(i).as_bytes(), r1.qual);
                assert_eq!(seq2.value(i).as_bytes(), r2.seq);
                assert_eq!(qual2.value(i).as_bytes(), r2.qual);
                *per_reason.entry(reasons.value(i).to_string()).or_insert(0) += 1;
            }
            rows += batch.num_rows();
        }

        assert!(rows > 0, "this mix is meant to drop reads");
        assert_eq!(rows, stats.total_dropped_reads());
        for (reason, count) in stats.dropped_by_frequency() {
            assert_eq!(
                per_reason.get(reason).copied().unwrap_or(0),
                count,
                "{reason}"
            );
        }
    }
}
