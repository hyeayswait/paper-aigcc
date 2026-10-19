"""T7 单元测试：多系统分数台账（paper_aigc/io/score_ledger.py）。

所有临时文件都写到 tempfile.mkdtemp() 下，不往仓库 data/ 目录丢文件。
覆盖：默认路径、add/load 往返、自动建父目录、自动补时间、文件不存在、
JSON 损坏、to_meta 的 rows 完整性与 latest 按 system+metric 取最新。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import unittest

from paper_aigc.io.score_ledger import (
    LedgerRecord,
    add_record,
    ledger_path,
    load_records,
    to_meta,
)

TIME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")


class _TempDirMixin:
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="paperaigc_ledger_")
        # 直接指定文件路径的场景用 self.path
        self.path = os.path.join(self.tmp, "score_ledger.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _add(self, time, version, system, metric, value, note="", path=None):
        record = LedgerRecord(
            time=time,
            version=version,
            system=system,
            metric=metric,
            value=value,
            note=note,
        )
        return add_record(record, path=path or self.path)


class LedgerPathTest(_TempDirMixin, unittest.TestCase):
    def test_default_path_is_data_score_ledger_json(self) -> None:
        path = ledger_path()
        self.assertTrue(os.path.isabs(path))
        self.assertTrue(
            path.endswith(os.path.join("data", "score_ledger.json")), path
        )
        # 默认路径应位于项目根下（本测试文件上溯三层即项目根）
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        self.assertEqual(
            os.path.normcase(path),
            os.path.normcase(
                os.path.join(project_root, "data", "score_ledger.json")
            ),
        )

    def test_base_dir_overrides_project_root(self) -> None:
        expected = os.path.join(self.tmp, "data", "score_ledger.json")
        self.assertEqual(
            os.path.normcase(ledger_path(self.tmp)),
            os.path.normcase(os.path.abspath(expected)),
        )


class AddAndLoadTest(_TempDirMixin, unittest.TestCase):
    def test_add_three_records_then_load_three(self) -> None:
        self._add("2026-10-01 09:00", "V5", "朱雀", "疑似AI", 41.2, "初稿")
        self._add("2026-10-02 10:30", "V6", "PaperPass", "总复制比", 18.6)
        self._add("2026-10-03 11:45", "V6", "本工具", "AI特征", 26.5, "终稿")

        records = load_records(self.path)
        self.assertEqual(len(records), 3)
        self.assertEqual(
            [r.version for r in records], ["V5", "V6", "V6"]
        )
        self.assertEqual(
            [r.system for r in records], ["朱雀", "PaperPass", "本工具"]
        )
        self.assertEqual(
            [r.metric for r in records], ["疑似AI", "总复制比", "AI特征"]
        )
        self.assertEqual([r.value for r in records], [41.2, 18.6, 26.5])
        self.assertEqual(records[0].note, "初稿")
        self.assertEqual(records[1].note, "")
        first = records[0]
        self.assertIsInstance(first.value, float)

    def test_add_appends_instead_of_overwriting(self) -> None:
        self._add("2026-10-01 09:00", "V5", "知网", "总复制比", 20.0)
        self._add("2026-10-02 09:00", "V6", "知网", "总复制比", 15.5)
        records = load_records(self.path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[-1].value, 15.5)

    def test_add_creates_parent_directories(self) -> None:
        nested = os.path.join(self.tmp, "a", "b", "c", "ledger.json")
        self.assertFalse(os.path.exists(os.path.dirname(nested)))
        returned = self._add(
            "2026-10-01 09:00", "V6", "朱雀", "疑似AI", 30.0, path=nested
        )
        self.assertTrue(os.path.isfile(nested))
        self.assertEqual(
            os.path.normcase(returned), os.path.normcase(os.path.abspath(nested))
        )
        self.assertEqual(len(load_records(nested)), 1)

    def test_add_default_ledger_path_creates_data_dir(self) -> None:
        target = add_record(
            LedgerRecord(
                time="2026-10-01 12:00",
                version="V6",
                system="朱雀",
                metric="疑似AI",
                value=33.3,
            ),
            path=ledger_path(self.tmp),
        )
        self.assertTrue(os.path.isfile(target))
        self.assertTrue(
            os.path.isdir(os.path.join(self.tmp, "data")),
            "add_record 应自动创建 data 父目录",
        )
        # 清理，避免 tmp 删除后残留引用问题（tempfile 目录整体删除即可）
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_add_fills_missing_time(self) -> None:
        add_record(
            LedgerRecord(version="V6", system="朱雀", metric="疑似AI", value=12.3),
            path=self.path,
        )
        records = load_records(self.path)
        self.assertEqual(len(records), 1)
        self.assertRegex(records[0].time, TIME_PATTERN)

    def test_note_defaults_to_empty(self) -> None:
        record = LedgerRecord(
            time="2026-10-01 09:00",
            version="V6",
            system="知网",
            metric="总复制比",
            value=10.0,
        )
        self.assertEqual(record.note, "")

    def test_value_string_is_coerced_to_float(self) -> None:
        add_record(
            LedgerRecord(
                time="2026-10-01 09:00",
                version="V6",
                system="知网",
                metric="总复制比",
                value="17.85",
            ),
            path=self.path,
        )
        self.assertEqual(load_records(self.path)[0].value, 17.85)

    def test_written_file_is_utf8_json_list(self) -> None:
        self._add("2026-10-01 09:00", "V6", "朱雀", "疑似AI", 34.47)
        with open(self.path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self.assertIsInstance(raw, list)
        self.assertEqual(raw[0]["system"], "朱雀")
        self.assertEqual(raw[0]["value"], 34.47)
        self.assertEqual(
            set(raw[0]), {"time", "version", "system", "metric", "value", "note"}
        )


class LoadToleranceTest(_TempDirMixin, unittest.TestCase):
    def test_missing_file_returns_empty(self) -> None:
        missing = os.path.join(self.tmp, "not_exists.json")
        self.assertEqual(load_records(missing), [])

    def test_corrupt_json_returns_empty(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("{not valid json!!!")
        self.assertEqual(load_records(self.path), [])

    def test_non_list_json_returns_empty(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"foo": "bar"}, fh)
        self.assertEqual(load_records(self.path), [])

    def test_mixed_entries_skip_invalid(self) -> None:
        payload = [
            {"time": "2026-10-01 09:00", "version": "V6", "system": "朱雀",
             "metric": "疑似AI", "value": 20.0, "note": ""},
            "garbage",
            123,
        ]
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        records = load_records(self.path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].system, "朱雀")

    def test_corrupt_file_does_not_block_add(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("]]][[")
        add_record(
            LedgerRecord(
                time="2026-10-01 09:00",
                version="V6",
                system="知网",
                metric="总复制比",
                value=9.9,
            ),
            path=self.path,
        )
        records = load_records(self.path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].value, 9.9)


class ToMetaTest(_TempDirMixin, unittest.TestCase):
    def _records(self):
        return [
            LedgerRecord("2026-10-01 09:00", "V6", "朱雀", "疑似AI", 41.2, "一稿"),
            LedgerRecord("2026-10-03 09:00", "V6", "朱雀", "疑似AI", 28.3, "三稿"),
            LedgerRecord("2026-10-02 09:00", "V5", "朱雀", "疑似AI", 35.0),
            LedgerRecord("2026-10-01 10:00", "V6", "PaperPass", "总复制比", 19.8),
            LedgerRecord("2026-10-02 10:00", "V6", "PaperPass", "总复制比", 12.4),
            LedgerRecord("2026-10-01 11:00", "V6", "本工具", "AI特征", 22.0),
            LedgerRecord("2026-10-01 12:00", "V6", "本工具", "总复制比", 8.0),
        ]

    def test_rows_complete_and_in_order(self) -> None:
        records = self._records()
        meta = to_meta(records)
        self.assertEqual(len(meta["rows"]), len(records))
        for row, record in zip(meta["rows"], records):
            self.assertEqual(
                set(row),
                {"time", "version", "system", "metric", "value", "note"},
            )
            self.assertEqual(row["time"], record.time)
            self.assertEqual(row["version"], record.version)
            self.assertEqual(row["system"], record.system)
            self.assertEqual(row["metric"], record.metric)
            self.assertEqual(row["value"], record.value)
            self.assertEqual(row["note"], record.note)

    def test_latest_per_system_and_metric(self) -> None:
        meta = to_meta(self._records())
        latest = meta["latest"]

        self.assertEqual(
            latest["朱雀"]["疑似AI"],
            {"value": 28.3, "version": "V6", "time": "2026-10-03 09:00"},
        )
        self.assertEqual(
            latest["PaperPass"]["总复制比"],
            {"value": 12.4, "version": "V6", "time": "2026-10-02 10:00"},
        )
        self.assertEqual(
            latest["本工具"]["AI特征"],
            {"value": 22.0, "version": "V6", "time": "2026-10-01 11:00"},
        )
        # 同一 system 下多个 metric 都要保留
        self.assertEqual(set(latest["本工具"]), {"AI特征", "总复制比"})
        self.assertEqual(latest["本工具"]["总复制比"]["value"], 8.0)

    def test_latest_uses_latest_time_not_order(self) -> None:
        # 后写入的记录时间更早，latest 仍应取时间最新的那条
        records = [
            LedgerRecord("2026-10-05 10:00", "V6", "朱雀", "疑似AI", 50.0),
            LedgerRecord("2026-10-01 10:00", "V5", "朱雀", "疑似AI", 20.0),
        ]
        meta = to_meta(records)
        self.assertEqual(meta["latest"]["朱雀"]["疑似AI"]["value"], 50.0)
        self.assertEqual(meta["latest"]["朱雀"]["疑似AI"]["version"], "V6")

    def test_latest_same_time_keeps_later_record(self) -> None:
        records = [
            LedgerRecord("2026-10-01 10:00", "V5", "知网", "总复制比", 20.0),
            LedgerRecord("2026-10-01 10:00", "V6", "知网", "总复制比", 15.0),
        ]
        meta = to_meta(records)
        self.assertEqual(meta["latest"]["知网"]["总复制比"]["value"], 15.0)
        self.assertEqual(meta["latest"]["知网"]["总复制比"]["version"], "V6")

    def test_empty_records(self) -> None:
        self.assertEqual(to_meta([]), {"rows": [], "latest": {}})

    def test_meta_from_file_after_add(self) -> None:
        self._add("2026-10-01 09:00", "V5", "朱雀", "疑似AI", 40.0)
        self._add("2026-10-02 09:00", "V6", "朱雀", "疑似AI", 30.0)
        meta = to_meta(load_records(self.path))
        self.assertEqual(len(meta["rows"]), 2)
        self.assertEqual(meta["latest"]["朱雀"]["疑似AI"]["value"], 30.0)
        self.assertEqual(meta["latest"]["朱雀"]["疑似AI"]["version"], "V6")
        self.assertEqual(meta["latest"]["朱雀"]["疑似AI"]["time"], "2026-10-02 09:00")


if __name__ == "__main__":
    unittest.main()
