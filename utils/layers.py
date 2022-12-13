import torch
from torch import nn

""" Stereo/Parallax Attention Module
Idea described in the following paper: Wang, L. et al. Parallax Attention for Unsupervised Stereo Correspondence Learning. TPAMI 2020 
"""


class PAM(nn.Module):
    def __init__(self, in_channels, interim_channels):
        super().__init__()

        self.resblock = ResBlock(in_channels=in_channels, interim_channels=interim_channels)

        self.Wq = nn.Conv2d(in_channels=interim_channels, out_channels=interim_channels, kernel_size=1,
                            stride=1, padding=0)
        self.Wk = nn.Conv2d(in_channels=interim_channels, out_channels=interim_channels, kernel_size=1,
                            stride=1, padding=0)

        self.softmax = nn.Softmax(-1)

        self.output_conv = nn.Conv2d(in_channels=(in_channels + interim_channels), out_channels=interim_channels,
                                     kernel_size=1, stride=1, padding=0)

    def forward(self, dp_left, dp_right):
        n, c, h, w = dp_left.shape

        dp_left_proc = self.resblock(dp_left)
        dp_right_proc = self.resblock(dp_right)

        Q = self.Wq(dp_left_proc).permute(0, 2, 3, 1)  # n x c x h x w -> nh x w x c
        K = self.Wk(dp_right_proc).permute(0, 2, 3, 1)  # n x c x h x w -> nh x c x w

        # right to left
        attn_score = torch.bmm(Q.contiguous().view(-1, w, c), K.contiguous().view(-1, c, w))  # nh x w x w
        pam_right_to_left = self.softmax(attn_score)

        # left to right
        attn_score_T = attn_score.permute(0, 2, 1)
        pam_left_to_right = self.softmax(attn_score_T)

        dp_right_temp = dp_right.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # nh x w x c
        dp_left_warped = torch.bmm(pam_right_to_left, dp_right_temp).contiguous().view(n, h, w, c).permute(0, 3, 1,
                                                                                                           2)  # back to n x c x h x w

        dp_left_temp = dp_left.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # nh x w x c
        dp_right_warped = torch.bmm(pam_left_to_right, dp_left_temp).contiguous().view(n, h, w, c).permute(0, 3, 1,
                                                                                                           2)  # back to n x c x h x w

        out_dp_left = self.output_conv(torch.cat((dp_left_warped, dp_left), dim=1))
        out_dp_right = self.output_conv(torch.cat((dp_right_warped, dp_right), dim=1))

        return out_dp_left, out_dp_right


class PAM3(nn.Module):
    def __init__(self, in_channels, interim_channels):
        super().__init__()

        self.convblock = ConvBlock(in_channels=in_channels, interim_channels=interim_channels)

        self.Wq = nn.Conv2d(in_channels=interim_channels, out_channels=interim_channels, kernel_size=1,
                            stride=1, padding=0)
        self.Wk = nn.Conv2d(in_channels=interim_channels, out_channels=interim_channels, kernel_size=1,
                            stride=1, padding=0)

        self.softmax = nn.Softmax(-1)

        self.output_conv = nn.Conv2d(in_channels=(in_channels + interim_channels), out_channels=interim_channels,
                                     kernel_size=1, stride=1, padding=0)

    def forward(self, dp_left, dp_right):
        n, c, h, w = dp_left.shape

        dp_left_proc = self.resblock(dp_left)
        dp_right_proc = self.resblock(dp_right)

        Q = self.Wq(dp_left_proc).permute(0, 2, 3, 1)  # n x c x h x w -> nh x w x c
        K = self.Wk(dp_right_proc).permute(0, 2, 3, 1)  # n x c x h x w -> nh x c x w

        # right to left
        attn_score = torch.bmm(Q.contiguous().view(-1, w, c), K.contiguous().view(-1, c, w))  # nh x w x w
        pam_right_to_left = self.softmax(attn_score)

        # left to right
        attn_score_T = attn_score.permute(0, 2, 1)
        pam_left_to_right = self.softmax(attn_score_T)

        dp_right_temp = dp_right.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # nh x w x c
        dp_left_warped = torch.bmm(pam_right_to_left, dp_right_temp).contiguous().view(n, h, w, c).permute(0, 3, 1, 2)  # back to n x c x h x w

        dp_left_temp = dp_left.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # nh x w x c
        dp_right_warped = torch.bmm(pam_left_to_right, dp_left_temp).contiguous().view(n, h, w, c).permute(0, 3, 1, 2)  # back to n x c x h x w

        out_dp_left = self.output_conv(torch.cat((dp_left_warped, dp_left), dim=1))
        out_dp_right = self.output_conv(torch.cat((dp_right_warped, dp_right), dim=1))

        return out_dp_left, out_dp_right


class PAM2(nn.Module):
    def __init__(self, in_channels, interim_channels):
        super().__init__()

        self.resblock = ResBlock(in_channels=in_channels, interim_channels=interim_channels)

        self.Wq = nn.Conv2d(in_channels=interim_channels, out_channels=interim_channels, kernel_size=1,
                            stride=1, padding=0)
        self.Wk = nn.Conv2d(in_channels=interim_channels, out_channels=interim_channels, kernel_size=1,
                            stride=1, padding=0)

        self.softmax = nn.Softmax(-1)

        self.output_conv1 = nn.Conv2d(in_channels=(in_channels + interim_channels), out_channels=interim_channels,
                                      kernel_size=1, stride=1, padding=0)
        self.output_relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.output_conv1_1 = nn.Conv2d(in_channels=2 * interim_channels, out_channels=2 * interim_channels,
                                        kernel_size=1, stride=1, padding=0)

    def forward(self, dp_left, dp_right):
        n, c, h, w = dp_left.shape

        dp_left_proc = self.resblock(dp_left)
        dp_right_proc = self.resblock(dp_right)

        Q = self.Wq(dp_left_proc).permute(0, 2, 3, 1)  # n x c x h x w -> nh x w x c
        K = self.Wk(dp_right_proc).permute(0, 2, 3, 1)  # n x c x h x w -> nh x c x w

        # right to left
        attn_score = torch.bmm(Q.contiguous().view(-1, w, c), K.contiguous().view(-1, c, w))  # nh x w x w
        pam_right_to_left = self.softmax(attn_score)

        # left to right
        attn_score_T = attn_score.permute(0, 2, 1)
        pam_left_to_right = self.softmax(attn_score_T)

        dp_right_temp = dp_right.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # nh x w x c
        dp_left_warped = torch.bmm(pam_right_to_left, dp_right_temp).contiguous().view(n, h, w, c).permute(0, 3, 1, 2)  # back to n x c x h x w

        dp_left_temp = dp_left.permute(0, 2, 3, 1).contiguous().view(-1, w, c)  # nh x w x c
        dp_right_warped = torch.bmm(pam_left_to_right, dp_left_temp).contiguous().view(n, h, w, c).permute(0, 3, 1, 2)  # back to n x c x h x w

        out_dp_left = self.output_relu1(self.output_conv1(torch.cat((dp_left_warped, dp_left), dim=1)))
        out_dp_right = self.output_relu1(self.output_conv1(torch.cat((dp_right_warped, dp_right), dim=1)))

        merged_output = self.output_conv1_1(torch.cat((out_dp_left, out_dp_right), dim=1))

        return merged_output


class ResBlock(nn.Module):
    def __init__(self, in_channels, interim_channels):
        super().__init__()

        # TODO: assuming in_channels == interim_channels, which is true currently
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=interim_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=interim_channels, out_channels=interim_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, input):
        output1 = self.resblock(input)
        output = output1 + input

        return output


class ConvBlock(nn.Module):
    def __init__(self, in_channels, interim_channels):
        super().__init__()

        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=interim_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=interim_channels, out_channels=interim_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, input):
        output = self.convblock(input)

        return output
