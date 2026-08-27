"""对照 RNN 和 LSTM 的序列输出与隐藏状态。"""

import torch
from torch import nn


def main() -> None:
    torch.manual_seed(42)
    batch_size, sequence_length, input_size = 3, 5, 8
    hidden_size = 16
    x = torch.randn(batch_size, sequence_length, input_size)

    rnn = nn.RNN(input_size, hidden_size, batch_first=True)
    lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    rnn_output, rnn_hidden = rnn(x)
    lstm_output, (lstm_hidden, cell_state) = lstm(x)

    print("输入 [B,T,D]：", x.shape)
    print("RNN 输出 / h_n：", rnn_output.shape, rnn_hidden.shape)
    print("LSTM 输出 / h_n / c_n：", lstm_output.shape, lstm_hidden.shape, cell_state.shape)

    # 最后一层、最后时刻的 output 与 h_n 表示同一隐藏状态。
    torch.testing.assert_close(rnn_output[:, -1], rnn_hidden[-1])
    torch.testing.assert_close(lstm_output[:, -1], lstm_hidden[-1])


if __name__ == "__main__":
    main()
