"""
DPDNet model definition according to the architecture description in the paper:
"Defocus Deblurring Using Dual-Pixel Data, ECCV 2020"
"""
import torch
import torch.nn as nn


def upsample(feature_map, scale=(2, 2), mode='nearest'):
    output = nn.Upsample(scale_factor=scale, mode=mode)(feature_map)
    return output


class DPDNet(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super().__init__()

        # encoder layer 1
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # encoder layer 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # encoder layer 3
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # encoder layer 4
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.drop4 = nn.Dropout2d(p=dropout_rate)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # encoder layer 5
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv5_1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.drop5 = nn.Dropout2d(p=dropout_rate)

        # decoder layer 1
        self.upsample6 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.upsample6_relu = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv6_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.relu6_1 = nn.ReLU(inplace=True)
        # decoder layer 2
        self.upsample7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.upsample7_relu = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv7_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.relu7_1 = nn.ReLU(inplace=True)
        # decoder layer 3
        self.upsample8 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.upsample8_relu = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.ReLU(inplace=True)
        self.conv8_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.relu8_1 = nn.ReLU(inplace=True)
        # decoder layer 4
        self.upsample9 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.upsample9_relu = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv9_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu9_1 = nn.ReLU(inplace=True)
        # final output layer
        self.conv9_2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu9_2 = nn.ReLU(inplace=True)
        self.conv9_3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.sig = nn.Sigmoid()

        # initialize weights
        self.apply(self._init_weight)

    def forward(self, dp_left, dp_right):
        input = torch.cat((dp_left, dp_right), dim=1)

        # encoder layer 1
        output1 = self.relu1_1(self.conv1_1(self.relu1(self.conv1(input))))
        pooled_output1 = self.pool1(output1)
        # encoder layer 2
        output2 = self.relu2_1(self.conv2_1(self.relu2(self.conv2(pooled_output1))))
        pooled_output2 = self.pool1(output2)
        # encoder layer 3
        output3 = self.relu3_1(self.conv3_1(self.relu3(self.conv3(pooled_output2))))
        pooled_output3 = self.pool1(output3)
        # encoder layer 4
        output4 = self.relu4_1(self.conv4_1(self.relu4(self.conv4(pooled_output3))))
        drop_output4 = self.drop4(output4)
        pooled_output4 = self.pool1(drop_output4)
        # encoder layer 5
        output5 = self.relu5_1(self.conv5_1(self.relu5(self.conv5(pooled_output4))))
        drop_output5 = self.drop5(output5)

        # decoder layer 1
        upsampled6_output = self.upsample6_relu(self.upsample6(upsample(drop_output5)))
        merged6 = torch.cat((upsampled6_output, drop_output4), dim=1)
        output6 = self.relu6_1(self.conv6_1(self.relu6(self.conv6(merged6))))
        # decoder layer 2
        upsampled7_output = self.upsample7_relu(self.upsample7(upsample(output6)))
        merged7 = torch.cat((upsampled7_output, output3), dim=1)
        output7 = self.relu7_1(self.conv7_1(self.relu7(self.conv7(merged7))))
        # decoder layer 3
        upsampled8_output = self.upsample8_relu(self.upsample8(upsample(output7)))
        merged8 = torch.cat((upsampled8_output, output2), dim=1)
        output8 = self.relu8_1(self.conv8_1(self.relu8(self.conv8(merged8))))
        # decoder layer 4
        upsampled9_output = self.upsample9_relu(self.upsample9(upsample(output8)))
        merged9 = torch.cat((upsampled9_output, output1), dim=1)
        output9 = self.relu9_1(self.conv9_1(self.relu9(self.conv9(merged9))))
        # final output
        final_output = self.sig(self.conv9_3(self.relu9_2(self.conv9_2(output9))))

        return final_output

    def _init_weight(self, module):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # print(m)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
