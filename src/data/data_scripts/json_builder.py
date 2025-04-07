# ddro/datasets/json_builder.py
# coding=utf-8

import json
from dataclasses import dataclass
from io import BytesIO

import pyarrow as pa
import pyarrow.json as paj
import datasets


@dataclass
class JsonConfig(datasets.BuilderConfig):
    """BuilderConfig for JSON."""

    features: datasets.Features = None
    field: str = None
    use_threads: bool = True
    block_size: int = None
    newlines_in_values: bool = None

    @property
    def pa_read_options(self):
        return paj.ReadOptions(use_threads=self.use_threads, block_size=self.block_size)

    @property
    def pa_parse_options(self):
        return paj.ParseOptions(explicit_schema=self.schema, newlines_in_values=self.newlines_in_values)

    @property
    def schema(self):
        return pa.schema(self.features.type) if self.features is not None else None


class Json(datasets.ArrowBasedBuilder):
    BUILDER_CONFIG_CLASS = JsonConfig

    def _info(self):
        return datasets.DatasetInfo(features=self.config.features)

    def _split_generators(self, dl_manager):
        """Handle string, list, and dict data files."""
        if not self.config.data_files:
            raise ValueError(f"At least one data file must be specified, but got data_files={self.config.data_files}")
        
        data_files = dl_manager.download_and_extract(self.config.data_files)
        files = data_files if isinstance(data_files, (list, tuple)) else [data_files]

        if isinstance(data_files, dict):
            splits = [
                datasets.SplitGenerator(name=split_name, gen_kwargs={"files": files})
                for split_name, files in data_files.items()
            ]
        else:
            splits = [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"files": files})]

        return splits

    def _generate_tables(self, files):
        for i, file in enumerate(files):
            pa_table = self._load_json_as_table(file)
            yield i, pa_table

    def _load_json_as_table(self, file):
        """Loads a JSON file as a PyArrow table."""
        if self.config.field is not None:
            with open(file, encoding="utf-8") as f:
                dataset = json.load(f)

            # Keep only the specified field
            dataset = dataset[self.config.field]

            # Handle list of dicts or dict of lists
            if isinstance(dataset, (list, tuple)):
                json_content = "\n".join(json.dumps(row) for row in dataset)
                pa_table = paj.read_json(
                    BytesIO(json_content.encode("utf-8")),
                    read_options=self.config.pa_read_options,
                    parse_options=self.config.pa_parse_options,
                )
            else:
                pa_table = pa.Table.from_pydict(mapping=dataset, schema=self.config.schema)
        else:
            pa_table = self._read_json_file(file)

        return pa_table

    def _read_json_file(self, file):
        """Read a JSON file directly using PyArrow."""
        try:
            return paj.read_json(
                file,
                read_options=self.config.pa_read_options,
                parse_options=self.config.pa_parse_options,
            )
        except pa.ArrowInvalid:
            with open(file, encoding="utf-8") as f:
                dataset = json.load(f)
            raise ValueError(
                f"Unable to read records from the JSON file at {file}. "
                f"Please specify the field containing the records using `field='XXX'`. "
                f"The JSON file contains the following fields: {list(dataset.keys())}."
            )
