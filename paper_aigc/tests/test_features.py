"""core/features.py 的单元测试：契约、边界、确定性与特征语义。"""
from __future__ import annotations

import math
import unittest

from paper_aigc.core import features
from paper_aigc.core.types import FeatureSet

EXPECTED_KEYS = [
    "sentence_len_cv",
    "avg_sentence_len",
    "short_sentence_ratio",
    "long_sentence_ratio",
    "clause_len_cv",
    "connective_density",
    "boilerplate_density",
    "parallelism_density",
    "hedge_density",
    "idiom_density",
    "num_ratio",
    "first_person_density",
    "passive_density",
    "char_ttr",
    "repetition_ratio",
    "char_entropy",
    "punct_diversity",
]

AI_TEXT = (
    "在当今社会，人工智能技术不仅是推动产业升级的重要引擎，而且是重塑教育生态的关键力量。"
    "一方面，智能技术能够有效提升教学效率，另一方面，它也可能在一定程度上加剧教育不公平。"
    "值得注意的是，相关研究普遍认为，技术应用既需要制度保障，又需要教师参与。"
    "首先，政策层面应当统筹推进基础设施建设，进一步完善相关标准与规范。"
    "其次，学校层面应当有效整合优质资源，建立健全教师培训的长效机制。"
    "此外，教师层面应当持续优化教学设计，把技术工具真正融入课堂实践，进而提升学生的核心素养。"
    "然而，当前研究仍存在明显不足，相关结论有待进一步验证，实践路径尚需系统梳理。"
    "综上所述，人工智能与教育的深度融合具有重要的理论意义和广阔的应用前景，"
    "同时也面临着诸多现实挑战，需要在今后的研究中不断探索与完善。"
)

HUMAN_TEXT = (
    "我带的两个班这学期试了新的作业办法。第一次交上来的东西很糟，"
    "三十七个人里有十四个直接复制了网上的答案。后来我把题目改成必须结合课堂实验数据，"
    "情况才慢慢好转。说到底，学生不是不会写，是不知道写什么值得写。"
)

VARIED_TEXT = (
    "春天来了，河边的柳树抽出新芽。夏天午后，蝉鸣穿过老旧的窗棂。"
    "秋风吹过稻田，农民忙着收割。冬夜漫长，炉火映着孩子的脸。"
    "海边的渔村，清晨总有咸湿的风。山里的学校，铃声回荡在云雾之间。"
)

REPEATED_TEXT = "人工智能技术正在改变教育。" * 6


def _finite_non_negative(values: dict) -> bool:
    for value in values.values():
        if not isinstance(value, float):
            return False
        if not math.isfinite(value) or value < 0:
            return False
    return True


class FeatureMetaTest(unittest.TestCase):
    """FEATURE_META 的键名、权重、锚点必须符合 SPEC 第 3 节。"""

    def test_meta_has_exactly_17_fixed_keys_in_order(self) -> None:
        self.assertEqual(len(features.FEATURE_META), 17)
        self.assertEqual(list(features.FEATURE_META.keys()), EXPECTED_KEYS)
        self.assertEqual(features.FEATURE_KEYS, EXPECTED_KEYS)

    def test_meta_fields_are_complete(self) -> None:
        for key, meta in features.FEATURE_META.items():
            self.assertIn(key, EXPECTED_KEYS)
            self.assertIsInstance(meta["name"], str)
            self.assertTrue(meta["name"])
            self.assertIsInstance(meta["weight"], float)
            self.assertGreater(meta["weight"], 0.0)
            self.assertIsInstance(meta["desc"], str)
            self.assertTrue(meta["desc"])

    def test_anchor_tables_are_ascending_and_bounded(self) -> None:
        for key, meta in features.FEATURE_META.items():
            anchors = meta["anchors"]
            self.assertIsInstance(anchors, list)
            self.assertGreaterEqual(len(anchors), 2, key)
            last_x = None
            for pair in anchors:
                self.assertEqual(len(pair), 2, key)
                x, y = float(pair[0]), float(pair[1])
                self.assertTrue(math.isfinite(x) and math.isfinite(y), key)
                if last_x is not None:
                    self.assertGreater(x, last_x, key)
                last_x = x
                self.assertGreaterEqual(y, 0.0, key)
                self.assertLessEqual(y, 1.0, key)


class ExtractFeaturesBoundaryTest(unittest.TestCase):
    """空文本、超短文本、纯英文、仅标点等边界必须安全且返回全 0。"""

    def test_empty_and_blank_return_all_zeros(self) -> None:
        for text in ("", "   ", "\n\t\r\n", "\u3000\u3000", "\ufeff"):
            values = features.extract_features_dict(text)
            self.assertEqual(list(values.keys()), EXPECTED_KEYS)
            for key in EXPECTED_KEYS:
                self.assertEqual(values[key], 0.0, (text, key))

    def test_punctuation_only_returns_all_zeros(self) -> None:
        for text in ("。", "！！！……？？", "，、；：", "——（）《》"):
            values = features.extract_features_dict(text)
            self.assertEqual(values, {key: 0.0 for key in EXPECTED_KEYS})

    def test_short_chinese_returns_all_zeros(self) -> None:
        values = features.extract_features_dict("你好。")
        self.assertEqual(values, {key: 0.0 for key in EXPECTED_KEYS})
        self.assertEqual(features.MIN_CJK_FOR_FEATURES, 5)

    def test_pure_english_does_not_raise_and_is_all_zero(self) -> None:
        values = features.extract_features_dict(
            "This is an English sentence. It contains no Chinese characters at all."
        )
        self.assertEqual(list(values.keys()), EXPECTED_KEYS)
        self.assertTrue(_finite_non_negative(values))
        self.assertEqual(values, {key: 0.0 for key in EXPECTED_KEYS})

    def test_weird_characters_do_not_raise(self) -> None:
        for text in ("\x00\x01abc", "😀😀😀😀😀", "ＡＢＣ１２３", "汉" * 3):
            values = features.extract_features_dict(text)
            self.assertEqual(list(values.keys()), EXPECTED_KEYS)
            self.assertTrue(_finite_non_negative(values))


class ExtractFeaturesContractTest(unittest.TestCase):
    """正常文本的返回值契约。"""

    def test_returns_feature_set_with_all_keys(self) -> None:
        result = features.extract_features(AI_TEXT)
        self.assertIsInstance(result, FeatureSet)
        self.assertEqual(list(result.values.keys()), EXPECTED_KEYS)
        self.assertTrue(_finite_non_negative(result.values))

    def test_extract_features_dict_is_a_copy(self) -> None:
        first = features.extract_features_dict(AI_TEXT)
        second = features.extract_features_dict(AI_TEXT)
        self.assertEqual(first, second)
        first["char_ttr"] = 999.0
        self.assertNotEqual(features.extract_features_dict(AI_TEXT)["char_ttr"], 999.0)
        self.assertEqual(
            features.extract_features_dict(AI_TEXT)["char_ttr"],
            second["char_ttr"],
        )

    def test_deterministic_across_calls(self) -> None:
        for text in (AI_TEXT, HUMAN_TEXT, VARIED_TEXT, REPEATED_TEXT):
            self.assertEqual(
                features.extract_features_dict(text),
                features.extract_features_dict(text),
            )

    def test_value_ranges_are_plausible(self) -> None:
        for text in (AI_TEXT, HUMAN_TEXT, VARIED_TEXT, REPEATED_TEXT * 4):
            values = features.extract_features_dict(text)
            for key in EXPECTED_KEYS:
                self.assertGreaterEqual(values[key], 0.0, (text, key))
            self.assertLessEqual(values["short_sentence_ratio"], 1.0)
            self.assertLessEqual(values["long_sentence_ratio"], 1.0)
            self.assertLessEqual(values["char_ttr"], 1.0)
            self.assertLessEqual(values["repetition_ratio"], 1.0)
            self.assertLessEqual(values["idiom_density"], 1.0)
            self.assertGreaterEqual(values["char_entropy"], 0.0)
            self.assertGreaterEqual(values["punct_diversity"], 0.0)

    def test_average_sentence_length(self) -> None:
        # 「今天天气很好。」6 字，「我们去公园散步。」7 字，均值 6.5
        values = features.extract_features_dict("今天天气很好。我们去公园散步。")
        self.assertAlmostEqual(values["avg_sentence_len"], 6.5, places=6)
        self.assertEqual(values["short_sentence_ratio"], 1.0)
        self.assertEqual(values["long_sentence_ratio"], 0.0)
        self.assertGreaterEqual(values["sentence_len_cv"], 0.0)

    def test_long_text_mattr_is_bounded(self) -> None:
        values = features.extract_features_dict(AI_TEXT * 8)
        self.assertGreater(values["char_ttr"], 0.0)
        self.assertLessEqual(values["char_ttr"], 1.0)
        self.assertGreater(values["repetition_ratio"], 0.0)


class FeatureSemanticsTest(unittest.TestCase):
    """各特征词表/统计方向应能被对应文本触发。"""

    def test_connective_density_detects_transitions(self) -> None:
        text = "首先，我们需要数据；其次，我们需要方法；因此，结论是可靠的。"
        self.assertGreater(features.extract_features_dict(text)["connective_density"], 0.0)

    def test_boilerplate_density_detects_cliche(self) -> None:
        text = "综上所述，该研究具有重要的理论意义和广阔的应用前景。"
        self.assertGreater(features.extract_features_dict(text)["boilerplate_density"], 0.0)

    def test_parallelism_density_detects_frames(self) -> None:
        text = "不仅要提高效率，而且要保证质量，同时还要兼顾公平。"
        self.assertGreater(features.extract_features_dict(text)["parallelism_density"], 0.0)

    def test_hedge_density_detects_hedges(self) -> None:
        text = "这可能是最好的办法，在一定程度上也是必要的，或许还更稳妥。"
        self.assertGreater(features.extract_features_dict(text)["hedge_density"], 0.0)

    def test_idiom_density_detects_four_char_words(self) -> None:
        text = "这项改革举足轻重，与民生息息相关，二者相辅相成。"
        self.assertGreater(features.extract_features_dict(text)["idiom_density"], 0.0)

    def test_num_ratio_detects_numbers(self) -> None:
        text = "2023 年共有 35% 的学校参与，有效样本 412 份。"
        self.assertGreater(features.extract_features_dict(text)["num_ratio"], 0.0)

    def test_first_person_density_detects_author(self) -> None:
        text = "我认为这个方法可行，我们可以再试一次，笔者对此比较乐观。"
        self.assertGreater(features.extract_features_dict(text)["first_person_density"], 0.0)

    def test_passive_density_detects_passive_markers(self) -> None:
        text = "这个问题受到广泛关注，相关机制得以完善，方案已被采纳。"
        self.assertGreater(features.extract_features_dict(text)["passive_density"], 0.0)

    def test_repetition_and_ttr_directions(self) -> None:
        repeated = features.extract_features_dict(REPEATED_TEXT)
        varied = features.extract_features_dict(VARIED_TEXT)
        self.assertGreater(repeated["repetition_ratio"], varied["repetition_ratio"])
        self.assertLess(repeated["char_ttr"], varied["char_ttr"])

    def test_ai_text_is_more_template_than_human_text(self) -> None:
        ai_values = features.extract_features_dict(AI_TEXT)
        human_values = features.extract_features_dict(HUMAN_TEXT)
        self.assertGreater(ai_values["connective_density"], human_values["connective_density"])
        self.assertGreater(ai_values["boilerplate_density"], human_values["boilerplate_density"])
        self.assertGreater(ai_values["avg_sentence_len"], human_values["avg_sentence_len"])
        self.assertGreater(ai_values["long_sentence_ratio"], human_values["long_sentence_ratio"])
        self.assertGreater(human_values["first_person_density"], ai_values["first_person_density"])


class RegressionFixTest(unittest.TestCase):
    """联调发现问题的回归测试：套话漏检、TTR 锚点、弱信号降权。"""

    def test_structural_boilerplate_is_detected(self) -> None:
        text = (
            "首先，需要明确研究问题。其次，需要梳理相关文献。"
            "此外，还需要说明研究方法。综上所述，本文旨在回答上述问题。"
        )
        values = features.extract_features_dict(text)
        self.assertGreaterEqual(values["boilerplate_density"], 2.0)

    def test_plain_ttr_separates_repeated_and_varied_paragraphs(self) -> None:
        repeated = (
            "人工智能技术正在改变教育模式，人工智能技术正在改变教育评价，"
            "人工智能技术正在改变教育管理，人工智能技术正在推动教育发展。"
        )
        varied = (
            "春天的柳树抽出新芽，夏夜的蝉鸣穿过窗棂。秋天的稻田翻起金浪，"
            "冬天的炉火映着孩子的脸。海边的渔村飘着咸湿的风，山里的学校回荡着钟声。"
        )
        repeated_values = features.extract_features_dict(repeated)
        varied_values = features.extract_features_dict(varied)
        self.assertLess(repeated_values["char_ttr"], varied_values["char_ttr"])

    def test_weak_signal_features_are_downweighted(self) -> None:
        meta = features.FEATURE_META
        self.assertLessEqual(float(meta["char_entropy"]["weight"]), 0.3)
        self.assertIn("弱信号", str(meta["char_entropy"]["desc"]))
        self.assertLess(float(meta["clause_len_cv"]["weight"]), 0.4)
        self.assertGreaterEqual(float(meta["boilerplate_density"]["weight"]), 1.0)


if __name__ == "__main__":
    unittest.main()
