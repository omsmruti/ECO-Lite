import torch
import torch.nn as nn
import torchvision.transforms as transforms
import csv   
import PIL.Image as Image
import cv2


def label_ids(labels_path):
    id_dict = {}
    with open(labels_path) as f:
        reader = csv.DictReader(f, delimiter=",", quotechar='"')
        for row in reader:
            id_dict.setdefault(row["label"], int(row["label_id"])-1)
    return id_dict


id_dict= label_ids(labels_path= 'kinetics_400_labels.csv')

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class conv_block_3d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block_3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(True)
    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))

class Inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_3x3_double, out_3x3_1, out_3x3_2, out_avg_pool):
        super(Inception_block, self).__init__()

        self.branch1 = conv_block(in_channels, out_1x1, kernel_size= 1, stride=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1, stride=1 ),
            conv_block(red_3x3, out_3x3, kernel_size=3, stride= 1, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_3x3_double, kernel_size=1, stride= 1),
            conv_block(red_3x3_double, out_3x3_1, kernel_size= 3, stride= 1,padding= 1),
            conv_block(out_3x3_1, out_3x3_2, kernel_size= 3, stride= 1, padding= 1)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, ceil_mode= True, count_include_pad= True), #ceil_mode=True or False??
            conv_block(in_channels, out_avg_pool, kernel_size=1, stride= 1)
        )
    def forward(self, x):
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


#Buliding the ECO-Lite Network

class ECO_Lite(nn.Module):
  def __init__(self):
    super(ECO_Lite, self).__init__()
    self.conv1= conv_block(in_channels= 3, out_channels= 64, kernel_size= 7, stride=2, padding=3)
    self.maxpool1= nn.MaxPool2d(kernel_size= 3, stride= 2, dilation= 1, ceil_mode= True)
    self.conv2= conv_block(64, 64, kernel_size= 1, stride= 1)
    self.conv3= conv_block(64, 192, kernel_size= 3, stride= 1, padding= 1)
    self.maxpool2= nn.MaxPool2d(kernel_size= 3, stride= 2, dilation= 1, ceil_mode= True)
    self.inception_3a= Inception_block(192, 64, 64, 64, 64, 96, 96, 32)
    self.inception_3b= Inception_block(256, 64, 64, 96, 64, 96, 96, 64)
    self.inception_3c= nn.Sequential(
        conv_block(320, 64, kernel_size=1, stride=1),
        conv_block(64, 96, kernel_size=3, stride=1, padding=1)
        )
    #### uptill here everything is correct ####
    # output is (n,96,28,28) where n is number of frames sampled from the video
    #3D Resnet-18

    self.res3a_2 = nn.Conv3d(96, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    self.res3a_bn = nn.BatchNorm3d(128)
    self.res3a_relu = nn.ReLU(inplace=True)
    self.res3b_1 = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    self.res3b_1_bn = nn.BatchNorm3d(128)
    self.res3b_1_relu = nn.ReLU(inplace=True)
    self.res3b_2 = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    self.res3b_bn = nn.BatchNorm3d(128)
    self.res3b_relu = nn.ReLU(inplace=True)

    self.res4a_1 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
    self.res4a_1_bn = nn.BatchNorm3d(256)
    self.res4a_1_relu = nn.ReLU(inplace=True)
    self.res4a_2 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    self.res4a_down = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
    self.res4a_bn = nn.BatchNorm3d(256)
    self.res4a_relu = nn.ReLU(inplace=True)
    self.res4b_1 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    self.res4b_1_bn = nn.BatchNorm3d(256)
    self.res4b_1_relu = nn.ReLU(inplace=True)
    self.res4b_2 = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    self.res4b_bn = nn.BatchNorm3d(256)
    self.res4b_relu = nn.ReLU(inplace=True)

    self.res5a_1 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
    self.res5a_1_bn = nn.BatchNorm3d(512)
    self.res5a_1_relu = nn.ReLU(inplace=True)
    self.res5a_2 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    self.res5a_down = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
    self.res5a_bn = nn.BatchNorm3d(512)
    self.res5a_relu = nn.ReLU(inplace=True)
    self.res5b_1 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    self.res5b_1_bn = nn.BatchNorm3d(512)
    self.res5b_1_relu = nn.ReLU(inplace=True)
    self.res5b_2 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    self.res5b_bn = nn.BatchNorm3d(512)
    self.res5b_relu = nn.ReLU(inplace=True)

    self.avg_pooling_layer= nn.AvgPool3d(kernel_size= (4,7,7), stride= 1)   #check this once
    self.fc1= nn.Linear(in_features= 512, out_features= 400)

  def forward(self, input):

    bs, ns, c, h, w = input.shape
    out = input.view(-1, c, h, w)

    conv1_7x7_out = self.conv1(out)
    pool1_3x3_s2_out = self.maxpool1(conv1_7x7_out)
    conv2_3x3_reduce_out = self.conv2(pool1_3x3_s2_out)
    conv2_3x3_out = self.conv3(conv2_3x3_reduce_out)
    pool2_3x3_s2_out = self.maxpool2(conv2_3x3_out)
        
    inception_3a_output_out = self.inception_3a(pool2_3x3_s2_out)       
    inception_3b_output_out = self.inception_3b(inception_3a_output_out)    
    inception_3c_output_out = self.inception_3c(inception_3b_output_out)

    #############################

    out1= inception_3c_output_out.view(-1, ns, 96, 28, 28)
    out2= torch.transpose(out1, 1, 2)

    residual_res3a_2 = self.res3a_2(out2)
    res3a_bn_out = self.res3a_bn(residual_res3a_2)
    res3a_relu_out = self.res3a_relu(res3a_bn_out)
    res3b_1_out = self.res3b_1(res3a_relu_out)
    res3b_1_bn_out = self.res3b_1_bn(res3b_1_out)
    res3b_relu_out = self.res3b_relu(res3b_1_bn_out)
    res3b_2_out = self.res3b_2(res3b_relu_out)
    res3b_2_out += residual_res3a_2
    res3b_bn_out = self.res3b_bn(res3b_2_out)
    res3b_relu_out = self.res3b_relu(res3b_bn_out)

    residual_res3b_relu_out = self.res4a_down(res3b_relu_out)
    res4a_1_out = self.res4a_1(res3b_relu_out)
    res4a_1_bn_out = self.res4a_1_bn(res4a_1_out)
    res4a_1_relu_out = self.res4a_1_relu(res4a_1_bn_out)
    res4a_2_out = self.res4a_2(res4a_1_relu_out)
    res4a_2_out += residual_res3b_relu_out
    residual2_res4a_2_out = res4a_2_out
    res4a_bn_out = self.res4a_bn(res4a_2_out)
    res4a_relu_out = self.res4a_relu(res4a_bn_out)
    res4b_1_out = self.res4b_1(res4a_relu_out)
    res4b_1_bn_out = self.res4b_1_bn(res4b_1_out)
    res4b_1_relu_out = self.res4b_1_relu(res4b_1_bn_out)
    res4b_2_out = self.res4b_2(res4b_1_relu_out)
    res4b_2_out += residual2_res4a_2_out
    res4b_bn_out = self.res4b_bn(res4b_2_out)
    res4b_relu_out = self.res4b_relu(res4b_bn_out)

    residual_res5a_down = self.res5a_down(res4b_relu_out)
    res5a_1_out = self.res5a_1(res4b_relu_out)
    res5a_1_bn_out = self.res5a_1_bn(res5a_1_out)
    res5a_1_relu_out = self.res5a_1_relu(res5a_1_bn_out)
    res5a_2_out = self.res5a_2(res5a_1_relu_out)
    res5a_2_out += residual_res5a_down  # res5a
    residual2_res5a_2_out = res5a_2_out
    res5a_bn_out = self.res5a_bn(res5a_2_out)
    res5a_relu_out = self.res5a_relu(res5a_bn_out)
    res5b_1_out = self.res5b_1(res5a_relu_out)
    res5b_1_bn_out = self.res5b_1_bn(res5b_1_out)
    res5b_1_relu_out = self.res5b_1_relu(res5b_1_bn_out)
    res5b_2_out = self.res5b_2(res5b_1_relu_out)
    res5b_2_out += residual2_res5a_2_out  
    res5b_bn_out = self.res5b_bn(res5b_2_out)
    res5b_relu_out = self.res5b_relu(res5b_bn_out)
    x1 = self.avg_pooling_layer(res5b_relu_out)
    x2 = self.fc1(x1.flatten())
    return x2

#loading the model
device = torch.device('cpu')
model_path= 'eco-lite-model.pth.tar'
model=ECO_Lite()
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


mean=[104, 117, 123]
std=[1, 1, 1]

image_transforms= transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

images=[]

def model_output(input_tensor):
    with torch.no_grad():
        prediction= model(input_tensor)
    return list(id_dict.keys())[list(id_dict.values()).index(int(torch.argmax(prediction)))]

from flask import Flask , Response , render_template

cam = cv2.VideoCapture(0)
app = Flask(__name__)

def stream():
    while True :
        ret,frame = cam.read()
        frame_edited= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_edited= Image.fromarray(frame_edited)
        frame_edited= image_transforms(frame_edited)
        frame_edited=frame_edited*255
        frame_edited=transforms.Normalize(mean, std)(frame_edited)
        images.append(frame_edited)
        if len(images)==16:
            input_to_model= torch.stack(images, dim=0).unsqueeze(dim=0)
            text= model_output(input_to_model)
            cv2.putText(frame, text, (10,80),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255) , thickness = 2 )
            del images[0]
        ret2, imgencode = cv2.imencode('.jpg',frame)
        strinData = imgencode.tobytes()
        yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+strinData+b'\r\n')

@app.route('/video')
def video():
    return Response(stream(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)











    
    
    
    
    
    