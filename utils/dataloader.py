import random
from typing import Dict, List, Optional, Union

import numpy as np
import torch


class CharDataLoader:
    """Data Loader for character network where the target is the next character in a sequence

    You can access the char_to_ix and ix_to_char maps after initializing by calling the
    dataloader.ix_to_char and dataloader.char_to_ix objects
    """

    def __init__(
        self,
        text_list: List[str],
        batch_size: int = 10,
        shuffle: bool = True,
        pad_char: str = "!",
        return_y: bool = True,
        ix_to_char: Optional[Dict[int, str]] = None,
    ):

        self.text_list = text_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            random.shuffle(self.text_list)
        self.pad_char = pad_char

        self.return_y = return_y

        chars = set([c for word in text_list for c in word])
        chars.add(self.pad_char)
        self.text_list = [text + self.pad_char for text in text_list]
        self.ix_to_char = ix_to_char
        if not self.ix_to_char:
            self.ix_to_char = {i: c for i, c in enumerate(chars)}
        self.char_to_ix = {c: i for i, c in self.ix_to_char.items()}

        self.n_batches = len(self.text_list) // self.batch_size
        if len(self.text_list) % self.batch_size != 0:
            self.n_batches += 1
        self.batch_num = 0

    def _text_to_ix(
        self, text: str, return_y: bool = True
    ) -> Union[List[int], Optional[List[int]]]:
        """Convert string to x and y lists of integers"""

        x = [self.char_to_ix[t] for t in text[:-1]]
        y = None
        if return_y:
            y = [self.char_to_ix[t] for t in text[1:]]
        return x, y

    def _build_batch(
        self, text: List[str], return_y: bool = True
    ) -> Union[torch.tensor, Optional[torch.tensor]]:
        """Build a batch of x and y tensors from list of strings"""
        x_batch = list()
        y_batch = list()

        max_chars = np.max([len(x) for x in text])
        padded_text = [t.ljust(max_chars, self.pad_char) for t in text]

        for text in padded_text:
            x_text, y_text = self._text_to_ix(text, return_y)
            x_batch.append(x_text)
            y_batch.append(y_text)
        x_batch = torch.tensor(x_batch)
        if return_y:
            y_batch = torch.tensor(y_batch)

        return x_batch, y_batch

    def __iter__(self):
        """Return self and reset batch count"""
        self.batch_num = 0
        return self

    def __next__(self):
        """Return batches for iterations"""
        if self.batch_num < self.n_batches:
            x_batch, y_batch = self._build_batch(
                text=self.text_list[
                    self.batch_num
                    * self.batch_size : (self.batch_num + 1)
                    * self.batch_size
                ],
                return_y=self.return_y,
            )
            self.batch_num += 1
            return x_batch, y_batch
        else:
            # If shuffling, reshuffle all text after all batches given
            if self.shuffle:
                random.shuffle(self.text_list)
            raise StopIteration
