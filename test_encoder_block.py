import unittest
import torch
from encoder_block import EncoderBlock


class TestEncoderBlock(unittest.TestCase):

    def setUp(self):
        """
        Setting up hyperparameters and EncoderBlock for testing.
        """
        # Hyperparameters
        self.d_model = 512
        self.num_heads = 8
        self.d_ff = 2048
        self.dropout = 0.1
        self.batch_size = 32
        self.seq_length = 10

        # EncoderBlock
        self.encoder_block = EncoderBlock(d_model=self.d_model,
                                          num_heads=self.num_heads,
                                          d_ff=self.d_ff,
                                          dropout=self.dropout)

    def test_forward_shape(self):
        """
        Check if the output tensor shape is correct after a forward pass.
        """
        sample_input = torch.randn(self.batch_size, self.seq_length, self.d_model)
        output = self.encoder_block(sample_input)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.d_model))

    def test_different_seqlength(self):
        """
        Test with a different sequence length.
        """
        different_seq_length = 20
        sample_input = torch.randn(self.batch_size, different_seq_length, self.d_model)
        output = self.encoder_block(sample_input)
        self.assertEqual(output.shape, (self.batch_size, different_seq_length, self.d_model))

    def test_different_inputdim(self):
        """
        Test with different input dimensions.
        """
        different_d_model = 256
        encoder_block = EncoderBlock(d_model=different_d_model,
                                     num_heads=self.num_heads,
                                     d_ff=self.d_ff,
                                     dropout=self.dropout)
        sample_input = torch.randn(self.batch_size, self.seq_length, different_d_model)
        output = encoder_block(sample_input)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, different_d_model))

    def test_masking(self):
        """
        Test with a mask.
        """
        mask = torch.tril(torch.ones(self.seq_length, self.seq_length))
        mask[mask == 0] = float('-inf')
        mask[mask == 1] = 0
        sample_input = torch.randn(self.batch_size, self.seq_length, self.d_model)
        output = self.encoder_block(sample_input, mask=mask)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.d_model))


if __name__ == "__main__":
    unittest.main()
