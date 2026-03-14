import pyarrow as pa
import pyarrow.parquet as pq


class ParquetShardWriter:
    def __init__(self, shard_dir, shard_prefix, shard_size, write_batch_size, logger):
        self.shard_dir = shard_dir
        self.shard_prefix = shard_prefix
        self.shard_size = shard_size
        self.write_batch_size = write_batch_size
        self.logger = logger

        self.shard_id = 0
        self.batch_id = 0

        self.current_writer = None
        self.current_schema = None
        self.current_shard_rows = 0
        self.current_shard_path = None

    def _open_new_writer(self, schema: pa.Schema):
        self.current_shard_path = self.shard_dir / f"{self.shard_prefix}_{self.shard_id:05d}.parquet"
        self.current_writer = pq.ParquetWriter(
            self.current_shard_path,
            schema=schema,
            compression="snappy"
        )
        self.current_schema = schema
        self.current_shard_rows = 0

    def _close_current_writer(self):
        if self.current_writer is not None:
            self.current_writer.close()
            self.logger.info(
                f"shard_completed | shard_id={self.shard_id:05d} | "
                f"rows={self.current_shard_rows}/{self.shard_size} | path={self.current_shard_path}"
            )
            self.current_writer = None
            self.current_schema = None
            self.current_shard_rows = 0
            self.current_shard_path = None
            self.shard_id += 1

    def save_results(self, results, force=False):
        if not force and len(results) < self.write_batch_size:
            return results

        if not results:
            if force:
                self._close_current_writer()
            return []

        pending = results

        while pending:
            if self.current_writer is None:
                room = self.shard_size
            else:
                room = self.shard_size - self.current_shard_rows

            if room <= 0:
                self._close_current_writer()
                room = self.shard_size

            chunk = pending[:room]
            pending = pending[room:]

            table = pa.Table.from_pylist(chunk)

            if self.current_writer is None:
                self._open_new_writer(table.schema)
            else:
                if table.schema != self.current_schema:
                    table = pa.Table.from_pylist(chunk, schema=self.current_schema)

            self.batch_id += 1
            self.logger.info(
                f"batch_flush | batch_id={self.batch_id} | shard_id={self.shard_id:05d} | "
                f"rows={len(chunk)} | shard_rows={self.current_shard_rows + len(chunk)}/{self.shard_size}"
            )

            self.current_writer.write_table(table)
            self.current_shard_rows += len(chunk)

            if self.current_shard_rows >= self.shard_size:
                self._close_current_writer()

        if force:
            self._close_current_writer()

        return []