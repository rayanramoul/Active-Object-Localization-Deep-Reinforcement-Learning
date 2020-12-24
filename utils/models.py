class FeatureExtractor(nn.Module):
      def __init__(self):
    super(FeatureExtractor, self).__init__() # recopier toute la partie convolutionnelle
    vgg16 = torchvision.models.vgg16(pretrained=True)
    vgg16.eval() # to not do dropout
    self.features = nn.Sequential( *list(vgg16.features.children()))
    # understand feature and classifier: https://www.kaggle.com/carloalbertobarbano/vgg16-transfer-learning-pytorch
    # garder une partie du classifieur, -2 pour s'arrêter à relu7
    self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-2])
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x
  
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear( in_features= 81 + 4096, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 1024, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features= 1024, out_features=9)
        )
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        return self.classifier(x)