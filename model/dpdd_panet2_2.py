import torch
import torch.nn as nn

from utils.layers import *


def upsample(feature_map, scale=(2, 2), mode='nearest'):
    output = nn.Upsample(scale_factor=scale, mode=mode)(feature_map)
    return output


class DPDD_PANet2_2(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super().__init__()

        # encoder layer 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pam1 = PAM3(in_channels=64, interim_channels=32)

        # encoder layer 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pam2 = PAM3(in_channels=128, interim_channels=128)

        # encoder layer 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pam3 = PAM3(in_channels=256, interim_channels=256)

        # encoder layer 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.drop4 = nn.Dropout2d(p=dropout_rate)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pam4 = PAM3(in_channels=512, interim_channels=256)

        # encoder layer 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )
        self.drop5 = nn.Dropout2d(p=dropout_rate)

        # decoder layer 1
        self.upsample6 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.upsample6_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv6 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv6_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu6_1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # decoder layer 2
        self.upsample7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.upsample7_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv7_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu7_1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # decoder layer 3
        self.upsample8 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.upsample8_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv8_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu8_1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # decoder layer 4
        self.upsample9 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.upsample9_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu9 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv9_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu9_1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # final output layer
        self.conv9_2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu9_2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv9_3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.sig = nn.Sigmoid()

        # initialize weights
        self.apply(self._init_weight)

    def forward(self, dp_left, dp_right):
        # encoder layer 1
        left_output1 = self.conv1(dp_left)
        left_pooled_output1 = self.pool1(left_output1)

        right_output1 = self.conv1(dp_right)
        right_pooled_output1 = self.pool1(right_output1)

        pam_left_output1, pam_right_output1 = self.pam1(left_pooled_output1, right_pooled_output1)

        # encoder layer 2
        left_output2 = self.conv2(pam_left_output1)
        left_pooled_output2 = self.pool2(left_output2)

        right_output2 = self.conv2(pam_right_output1)
        right_pooled_output2 = self.pool2(right_output2)

        pam_left_output2, pam_right_output2 = self.pam2(left_pooled_output2, right_pooled_output2)

        # encoder layer 3
        left_output3 = self.conv3(pam_left_output2)
        left_pooled_output3 = self.pool3(left_output3)

        right_output3 = self.conv3(pam_right_output2)
        right_pooled_output3 = self.pool3(right_output3)

        pam_left_output3, pam_right_output3 = self.pam3(left_pooled_output3, right_pooled_output3)

        # encoder layer 4
        left_output4 = self.conv4(pam_left_output3)
        left_drop_output4 = self.drop4(left_output4)
        left_pooled_output4 = self.pool4(left_drop_output4)

        right_output4 = self.conv4(pam_right_output3)
        right_drop_output4 = self.drop4(right_output4)
        right_pooled_output4 = self.pool4(right_drop_output4)

        pam_left_output4, pam_right_output4 = self.pam4(left_pooled_output4, right_pooled_output4)

        # encoder layer 5
        left_output5 = self.conv5(pam_left_output4)
        left_drop_output5 = self.drop5(left_output5)

        right_output5 = self.conv5(pam_right_output4)
        right_drop_output5 = self.drop5(right_output5)


        # decoder layer 1
        upsampled6_output = self.upsample6_relu(self.upsample6(upsample(left_drop_output5 + right_drop_output5)))
        merged6 = torch.cat((upsampled6_output, left_output4 + right_output4), dim=1)
        output6 = self.relu6_1(self.conv6_1(self.relu6(self.conv6(merged6))))
        # decoder layer 2
        upsampled7_output = self.upsample7_relu(self.upsample7(upsample(output6)))
        merged7 = torch.cat((upsampled7_output, left_output3 + right_output3), dim=1)
        output7 = self.relu7_1(self.conv7_1(self.relu7(self.conv7(merged7))))
        # decoder layer 3
        upsampled8_output = self.upsample8_relu(self.upsample8(upsample(output7)))
        merged8 = torch.cat((upsampled8_output, left_output2 + right_output2), dim=1)
        output8 = self.relu8_1(self.conv8_1(self.relu8(self.conv8(merged8))))
        # decoder layer 4
        upsampled9_output = self.upsample9_relu(self.upsample9(upsample(output8)))
        merged9 = torch.cat((upsampled9_output, left_output1 + right_output1), dim=1)
        output9 = self.relu9_1(self.conv9_1(self.relu9(self.conv9(merged9))))

        # final output
        final_output = self.sig(self.conv9_3(self.relu9_2(self.conv9_2(output9))))

        return final_output

    def _init_weight(self, module):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # print(m)
                nn.init.kaiming_normal_(tensor=m.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
