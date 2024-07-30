import torch
from torchvision.models.video import S3D_Weights, s3d

class TwoStreamS3D(torch.nn.Module):

    def __init__(self, num_classes: int=400, dropout: float=0.2):
        super(TwoStreamS3D, self).__init__()

        self.spatial_stream = s3d(weights=S3D_Weights.KINETICS400_V1, num_classes=400, dropout=dropout)
        self.temporal_stream = s3d(weights=S3D_Weights.KINETICS400_V1, num_classes=400, dropout=dropout)
        self.temporal_stream.classifier = torch.nn.Identity()

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True)
        )

    def forward(self, rgb_frames, optical_flow_frames):
        spatial_output = forward(self.spatial_stream, rgb_frames)
        temporal_output = forward(self.temporal_stream, optical_flow_frames)
        combined_output = torch.cat((spatial_output, temporal_output), dim=2)
        final_output = self.classifier(combined_output)
        final_output = torch.mean(final_output, dim=(2, 3, 4))

        return final_output
    
def forward(s3d_model, x):
    x = s3d_model.features(x)
    x = s3d_model.avgpool(x)
    return x
