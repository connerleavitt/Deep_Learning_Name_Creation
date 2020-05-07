import unittest

import numpy as np
from ddt import data, ddt, unpack

from utils.dataloader import CharDataLoader


@ddt
class DataLoaderTest(unittest.TestCase):
    @data(
        (
            ["ab", "cd"],
            {0: "a", 1: "b", 2: "c", 3: "d"},
            {"a": 0, "b": 1, "c": 2, "d": 3},
            "Test ix_to_char",
        )
    )
    @unpack
    def test_ix_to_char(self, text_list, ix_to_char, exp_return, test_description):
        dataloader = CharDataLoader(text_list, ix_to_char=ix_to_char)
        result = dataloader.char_to_ix
        self.assertDictEqual(exp_return, result, test_description)

    @data(
        (["test"], {0: "t", 1: "e", 2: "s"}, False, [0, 1, 2], "Test vector creation")
    )
    @unpack
    def test_text_to_ix(
        self, text_list, ix_to_char, return_y, exp_return, test_description
    ):
        dataloader = CharDataLoader(text_list, ix_to_char=ix_to_char, return_y=return_y)
        x, y = dataloader._text_to_ix(text_list[0], return_y)
        self.assertListEqual(exp_return, x, test_description)

    @data(
        (["test", "run", "for", "shuffle", "of", "data"], True, "Testing shuffle on"),
        (["test", "run", "for", "shuffle", "of", "data"], False, "Testing shuffle off"),
    )
    @unpack
    def test_shuffle(self, text_list, shuffle, test_description):
        dataloader = CharDataLoader(text_list, shuffle=shuffle)
        results = []
        for _ in range(2):
            for x, y in dataloader:
                results.append(x)

        res_1 = results[0].numpy()
        res_2 = results[1].numpy()

        if shuffle:
            self.assertFalse(np.allclose(res_1, res_2), test_description)
        else:
            self.assertTrue(np.allclose(res_1, res_2), test_description)

    @data(
        (["test"] * 1000, 10, "Test regular batch size"),
        (["test"] * 1000, 100, "Test large batch size"),
        (["test"] * 1000, 1, "Test single unit batch size"),
    )
    @unpack
    def test_batch_size(self, text_list, batch_size, test_description):
        dataloader = CharDataLoader(text_list, batch_size=batch_size)
        x, y = next(dataloader)
        self.assertTrue(x.shape[0] == batch_size, test_description)
        self.assertTrue(y.shape[0] == batch_size, test_description)


if __name__ == "__main__":
    unittest.main()
