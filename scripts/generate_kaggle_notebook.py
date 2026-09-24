import re

with open('notebooks/fanet_kaggle.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace MixPool
mixpool_regex = re.compile(r'class MixPool\(nn\.Module\):.*?return torch\.cat\(\[x1, x2\], dim=1\)', re.DOTALL)
new_mixpool = """class MixPool(nn.Module):
    \"\"\"Phase 7C: Learned Residual Decoupling\"\"\"
    def __init__(self, in_c, out_c):
        super().__init__()
        self.fmask = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, 1, 1, padding=0),
            nn.Sigmoid()
        )
        self.learned_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c // 2, 3, padding=1),
            nn.BatchNorm2d(out_c // 2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c // 2, 3, padding=1),
            nn.BatchNorm2d(out_c // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, m):
        fmask = self.fmask(x)
        m = nn.MaxPool2d((m.shape[2] // x.shape[2], m.shape[3] // x.shape[3]))(m)
        m_fg = m[:, 0:1].detach() # Sever the Feedback Trap
        
        combined = torch.cat([fmask, m_fg], dim=1)
        dynamic_gate = self.learned_gate(combined)
        
        keep = (dynamic_gate * m_fg) + fmask
        keep = torch.clamp(keep, 0.0, 1.0)
        
        x1 = self.conv1(x * keep)
        x2 = self.conv2(x)
        return torch.cat([x1, x2], dim=1)"""

content = mixpool_regex.sub(new_mixpool, content)

# Inject AdaptiveTverskyLoss before Loss definition
tversky_code = """class AdaptiveTverskyLoss(nn.Module):
    def __init__(self, smooth=1.0, alpha=0.7, beta=0.3):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        tp = torch.sum(y_true * y_pred, dim=[0, 2, 3])
        fp = torch.sum((1 - y_true) * y_pred, dim=[0, 2, 3])
        fn = torch.sum(y_true * (1 - y_pred), dim=[0, 2, 3])
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - torch.mean(tversky)

class DiceBCELoss(nn.Module):"""

content = content.replace('class DiceBCELoss(nn.Module):', tversky_code)

# Switch loss to Tversky in training block
content = content.replace('loss_fn = DiceBCELoss()', 'loss_fn = AdaptiveTverskyLoss(alpha=0.7, beta=0.3)')
content = content.replace('Loss:      DiceBCELoss', 'Loss:      Curriculum Adaptive Tversky (Phase 7C)')
content = content.replace('FANet Training & Evaluation Complete', 'FANet (Phase 7C) Training & Evaluation Complete')

with open('notebooks/fanet_phase7c_kaggle.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done!')
